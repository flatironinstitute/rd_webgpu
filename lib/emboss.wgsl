


struct parameters {
    size: f32,
    scalez: f32,
}

// Input and output panels interpreted as u32 rgba, assumed same shape.
@group(0) @binding(0) var<storage, read> inputBuffer : array<f32>;

@group(1) @binding(0) var<storage, read_write> outputBuffer : array<u32>;

@group(2) @binding(0) var<storage, read> parms: parameters;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3u) {
    let inputOffset = global_id.x;
    let side = u32(parms.size);
    let in_hw = vec2u(side, side);
    let in_location = panel_location_of(inputOffset, in_hw);
    if (in_location.is_valid) {
        let i = in_location.ij.x;
        let j = in_location.ij.y;
        let size = u32(parms.size);
        if (i < size && j < size) {
            let rightij = vec2u(i + 1, j);
            let upij = vec2u(i, j + 1);
            let rightlocation = panel_offset_of(rightij, in_hw);
            let uplocation = panel_offset_of(upij, in_hw);
            let right = inputBuffer[rightlocation.offset];
            let up = inputBuffer[uplocation.offset];
            let here = inputBuffer[in_location.offset];
            let scalez = parms.scalez;
            let p_here = vec3f(f32(i), f32(j), here * scalez);
            let p_right = vec3f(f32(i + 1), f32(j), right * scalez);
            let p_up = vec3f(f32(i), f32(j + 1), up * scalez);
            let normal = normalize(cross(p_right - p_here, p_up - p_here));
            let shift = 0.5 * (vec3f(1.0, 1.0, 1.0) + normal);
            let color = f_pack_color(abs(shift));
            outputBuffer[in_location.offset] = color;
        }
    }
}
