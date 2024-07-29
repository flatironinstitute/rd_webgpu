

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3u) {
    let outputOffset = global_id.x;
    let outputShape = outputDB.shape;
    let outputLocation = depth_buffer_indices(outputOffset, outputShape);
    if (outputLocation.valid) {
        // get the parameters for this location
        let DA = parms.DA;
        let DB = parms.DB;
        let dt = parms.dt;
        // get the values of k and f at this location
        let side = u32(outputDB.shape.height);
        let flocation = u32(outputLocation.ij.x);
        let klocation = u32(outputLocation.ij.y) + side;
        let f = inputBuffer[flocation];
        let k = inputBuffer[klocation];
        // initial values of A and B at this location
        let Aij = outputDB.data_and_depth[outputLocation.data_offset];
        let Bij = outputDB.data_and_depth[outputLocation.depth_offset];
        // next values of A and B at this location
        //var Aijnext = Aij;
        //var Bijnext = Bij;

        // something arbitrary
        Aijnext = Aij - Bij + k - f;
        Bijnext = -Aij + Bij + k + f;
        // Do the reaction diffusion update
        // xxx fill in the code here...

        // FOR DEBUGGING... just switch the values
        //Aijnext = Bij;
        //Bijnext = Aij;

        // FOR DEBUGGING... put k in A and f in B
        Aijnext = k;
        Bijnext = f;

        // for DEBUGGING... put DA in A and DB in B
        //Aijnext = DA;
        //Bijnext = DB;
        if (outputOffset == 0) {
            Aijnext = dt;
        }

        // write the updated values back to the depth buffer
        outputDB.data_and_depth[outputLocation.data_offset] = Aijnext;
        outputDB.data_and_depth[outputLocation.depth_offset] = Bijnext;
    }
}
