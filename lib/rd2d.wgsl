

// implement an in place update to a DepthBuffer that implements a reaction diffusion
// similar to
//
// https://github.com/benmaier/reaction-diffusion/blob/master/gray_scott.ipynb
//
// the "Value" portion of the depth buffer corresponds to the "A" array
// and the "Depth" corresponds to the "B" array
//

// Suffix
// Requires "depth_buffer.wgsl" and "panel_buffer.wgsl"

// array parameters for k and f implemented as a Nx2 panel
@group(0) @binding(0) var<storage, read> inputBuffer : array<f32>;

// arrays A and B implemented as a depth buffer
@group(1) @binding(0) var<storage, read_write> outputDB : DepthBufferF32;

struct parameters {
    DA: f32,  // diffusion rate of B
    DB: f32,  // diffusion rate of A
    dt: f32,  // time step
}

@group(2) @binding(0) var<storage, read> parms: parameters;

// define a linear workgroup size of length 256
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3u) {
    // output into the depth buffer
    let outputOffset = global_id.x; // output offset of this group (1D)
    let outputShape = outputDB.shape; 
    let outputLocation = depth_buffer_indices(outputOffset, outputShape);

    // after gotten the output location
    if (outputLocation.valid) {
        // get the parameters for this location
        let DA = parms.DA;
        let DB = parms.DB;
        let dt = parms.dt;

        // get the values of k and f at this location
        let side = u32(outputDB.shape.height);
        let flocation = u32(outputLocation.ij.x); // f goes horizontally
        let klocation = u32(outputLocation.ij.y) + side; // k goes vertically
        let f = inputBuffer[flocation];
        let k = inputBuffer[klocation];

        // initial values of A and B at this location
        var initAij = outputDB.data_and_depth[outputLocation.data_offset]; // A is the "value" portion
        var initBij = outputDB.data_and_depth[outputLocation.depth_offset]; // B is the "depth" portion

        // ------------------------- Calculating discrete laplacian -------------------------
        let i = outputLocation.ij.x;
        let j = outputLocation.ij.y;

        // get the four directional vectors
        let up = vec2i(i - 1, j);
        let down = vec2i(i + 1, j);
        let right = vec2i(i, j + 1);
        let left = vec2i(i, j - 1);

        // get the four required values to calculate
        let above = depth_buffer_location_of(up, outputShape);
        let below = depth_buffer_location_of(down, outputShape);
        let r = depth_buffer_location_of(right, outputShape);
        let l = depth_buffer_location_of(left, outputShape);

        // calculate the laplacian
        let LAij = -4 * initAij + outputDB.data_and_depth[above.data_offset]
        + outputDB.data_and_depth[below.data_offset]
        + outputDB.data_and_depth[r.data_offset]
        + outputDB.data_and_depth[l.data_offset];

        let LBij = -4 * initBij + outputDB.data_and_depth[above.depth_offset]
        + outputDB.data_and_depth[below.depth_offset]
        + outputDB.data_and_depth[r.depth_offset]
        + outputDB.data_and_depth[l.depth_offset];

        // ------------------------- Gray Scott Update -------------------------

        let diffAij = (DA * LAij - initAij * initBij * initAij * initBij + f * (f32(1) - initAij)) * dt;
        let diffBij = (DB * LBij + initAij * initBij * initAij * initBij - (k + f) * initBij) * dt;

        var Aijnext = initAij + diffAij;
        var Bijnext = initBij + diffBij;

        Aijnext = initBij;
        Bijnext = initAij;
        
        // if (outputOffset == 0) {
        //     Aijnext = dt;
        // }

        // write the updated values back to the depth buffer for JS to read
        outputDB.data_and_depth[outputLocation.data_offset] = Aijnext;
        outputDB.data_and_depth[outputLocation.depth_offset] = Bijnext;
    }
}
