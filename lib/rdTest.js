

export const name = "rdTest";

import * as webgpu_volume from "webgpu_volume";
import * as rdUpdate from "./rdUpdate.js";

function filler(size, min, max) {
    const result =  new Float32Array(size);
    const range = max - min;
    const delta = range / size;
    for (let i = 0; i < size; i++) {
        result[i] = min + i * delta;
    }
    return result;
}

export async function test() {
    const side = 4;
    const initialA = filler(side * side, 0.1, 0.2);
    const initialB = filler(side * side, 0.2, 0.3);
    const fArray = filler(side, 0.01, 0.02);
    const kArray = filler(side, 0.03, 0.04);
    const DA = 0.16;
    const DB = 0.08;
    const dt = 1.0;
    const updateRD = new rdUpdate.UpdateRD(
        side, 
        initialA, 
        initialB, 
        fArray, 
        kArray, 
        DA, 
        DB,
        dt);
    console.log("updateRD", updateRD);
    const context = new webgpu_volume.GPUContext.Context();
    await context.connect();
    updateRD.attach_to_context(context);
    updateRD.run();
    const results = await updateRD.pull_arrays();
    console.log("results", results);
};