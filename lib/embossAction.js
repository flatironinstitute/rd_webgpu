

import * as webgpu_volume from "webgpu_volume";
import emboss_wgsl from "./emboss.wgsl?raw";


class embossParameters extends webgpu_volume.GPUDataObject.DataObject {

    constructor(size, scalez) {
        super();
        this.size = size;
        this.scalez = scalez;
        this.buffer_size = 2 * Float32Array.BYTES_PER_ELEMENT;
    };
    load_buffer(buffer) {
        buffer = buffer || this.gpu_buffer;
        const arrayBuffer = buffer.getMappedRange();
        const mappedFloats = new Float32Array(arrayBuffer);
        mappedFloats[0] = this.size;
        mappedFloats[1] = this.scalez;
    };
}

export class EmbossAction extends webgpu_volume.UpdateAction.UpdateAction {
    constructor (
        scalez, // scaling factor for z values
        fromPanel, // panel to emboss containing float32 values
        toPanel, // panel to write u32 color embossed values
    ) {
        super();
        const size = fromPanel.width;
        // xxx panels should be square of same size.
        if (fromPanel.height !== size || toPanel.width !== size || toPanel.height !== size) {
            throw new Error("fromPanel and toPanel must have the same size");
        }
        this.parameters = new embossParameters(size, scalez);
        this.source = fromPanel;
        this.target = toPanel;
    };
    get_shader_module(context) {
        const panel_prefix = webgpu_volume.panel_buffer_wgsl;
        const shader = panel_prefix + emboss_wgsl;
        return context.device.createShaderModule({code: shader});
    };
    getWorkgroupCounts() {
        // loop through output
        return [Math.ceil(this.target.size / 256), 1, 1];
    }
}
