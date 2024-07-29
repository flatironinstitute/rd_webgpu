
import * as webgpu_volume from "webgpu_volume";
import rd2d from "./rd2d.wgsl?raw";

class rdParameters extends webgpu_volume.GPUDataObject.DataObject {

    constructor(DA, DB, dt) {
        super();
        this.DA = DA;
        this.DB = DB;
        this.dt = dt;
        this.buffer_size = 3 * Int32Array.BYTES_PER_ELEMENT;
    };
    load_buffer(buffer) {
        buffer = buffer || this.gpu_buffer;
        const arrayBuffer = buffer.getMappedRange();
        const mappedFloats = new Float32Array(arrayBuffer);
        mappedFloats[0] = this.DA;
        mappedFloats[1] = this.DB;
        mappedFloats[2] = this.dt;
    };
}

export class UpdateRD extends webgpu_volume.UpdateAction.UpdateAction {
    constructor( 
        side, // side length
        initialA, // initial A array of size side * side
        initialB,
        fArray, // feed rate for A array of size side
        kArray, // kill rate for B array of size side
        DA, // diffusion rate for A
        DB, // diffusion rate for B
        dt, // time step
    ) {
        super();
        const shape = [side, side];
        const updateDepthBuffer = new webgpu_volume.GPUDepthBuffer.DepthBuffer(
            shape,
            0, 0, // default depth and value (not used here)
            initialA,
            initialB,
            Float32Array, // data format
        )
        this.parameters = new rdParameters(DA, DB, dt);
        this.target = updateDepthBuffer;
        this.side = side;
        if (fArray.length !== side || kArray.length !== side) {
            throw new Error("fArray and kArray must have the same length as the depth buffer side");
        }
        const fkArray = new Float32Array(this.side * 2);
        fkArray.set(fArray);
        fkArray.set(kArray, this.side);
        this.fkArray = fkArray;
        this.source = new webgpu_volume.GPUColorPanel.Panel(this.side, 2);
    };
    async pull_arrays() {
        await this.target.pull_data();
        return {
            A: this.target.data,
            B: this.target.depths,
        };
    }
    get_shader_module(context) {
        const db_prefix = webgpu_volume.depth_buffer_wgsl;
        const pb_prefix = webgpu_volume.panel_buffer_wgsl;
        const gpu_shader = db_prefix + pb_prefix + rd2d;
        return context.device.createShaderModule({ code: gpu_shader });
    };
    getWorkgroupCounts() {
        // loop through output
        return [Math.ceil(this.target.size / 256), 1, 1];
    };
    attach_to_context(context) {
        super.attach_to_context(context);
        this.source.push_buffer(this.fkArray);
    };
}
