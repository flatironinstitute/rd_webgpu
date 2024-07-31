
import * as webgpu_volume from "webgpu_volume";
import * as rdUpdate from "./rdUpdate.js";
import * as embossAction from "./embossAction.js";

function dummyVolume() {
    const shape = [1,1,1];
    const data = new Uint32Array(1);
    return new webgpu_volume.GPUVolume.Volume(shape, data);
}

export class PipelineRD extends webgpu_volume.ViewVolume.View {
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
        super(dummyVolume());
        this.side = side;
        this.updateRD = new rdUpdate.UpdateRD(
            side, 
            initialA, 
            initialB, 
            fArray, 
            kArray, 
            DA, 
            DB,
            dt);
    };
    panel_sequence(context) {
        context = this.context || context;
        this.min_value = 0.0;
        this.max_value = 1.0;
        //this.updateRD.ofVolume.attach_to_context(context);
        this.updateRD.attach_to_context(context);
        const side = this.side;
        this.color_panel = context.panel(side, side);
        this.value_panel = context.panel(side, side);
        const depth_buffer = this.updateRD.target;
        this.flatten_action = depth_buffer.flatten_action(this.value_panel);
        this.gray_action = context.to_gray_panel(
            this.value_panel, this.color_panel, this.min_value, this.max_value);
        this.embossAction = new embossAction.EmbossAction(
            1.0, // scaling factor for z values
            this.value_panel, // panel to emboss containing float32 values
            this.color_panel, // panel to write u32 color embossed values
        );
        this.embossAction.attach_to_context(context);
        this.project_to_panel = context.sequence([
                this.updateRD, 
                this.flatten_action, 
                //this.gray_action, 
                this.embossAction,
            ]);
        return {
            sequence: this.project_to_panel,
            output_panel: this.color_panel,
        };
    }
}