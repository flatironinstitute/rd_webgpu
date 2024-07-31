

export const name = "rdTest";

import * as webgpu_volume from "webgpu_volume";
import * as rdUpdate from "./rdUpdate.js";
import * as rdPipeline from "./rdPipeline.js";

function filler(size, min, max) {
    const result = new Float32Array(size);
    const range = max - min;
    const delta = range / size;
    for (let i = 0; i < size; i++) {
        result[i] = min + i * delta;
    }
    return result;
}

function fillrandom(size, M, random_influence) {
    const result = new Float32Array(size);
    if (M === 'A') {
        for (let i = 0; i < size; i++) {
            result[i] = (1 - random_influence) + random_influence * Math.random();
        }
    } else if (M === 'B') {
        for (let i = 0; i < size; i++) {
            result[i] = random_influence * Math.random();
        }
    } else {
        console.log("cannot fill unknown array");
        return null;
    }
    return result
}

export async function test() {
    const side = 3;
    // min = 0.1, max = 0.2
    const initialA = filler(side * side, 1., 1.);
    initialA[4] = 0

    // min = 0.2, max = 0.3
    const initialB = filler(side * side, 1., 1.);
    initialB[4] = 0

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

export async function test_demo(canvas) {
    const side = 4;
    const initialA = filler(side * side, 0.1, 0.5);
    const initialB = filler(side * side, 0.5, 0.9);
    const fArray = filler(side, 0.01, 0.02);
    const kArray = filler(side, 0.03, 0.04);
    const DA = 0.16;
    const DB = 0.08;
    const dt = 1.0;
    const updateRD = new rdPipeline.PipelineRD(
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
    debugger;
    await context.connect();
    updateRD.ofVolume.attach_to_context(context);
    updateRD.attach_to_context(context);
    await updateRD.paint_on(canvas);
    //updateRD.run();
    return updateRD;
};

export async function test2() {
    const side = 100;
    const random_influence = 0.1;
    const initialA = fillrandom(side * side, 'A', random_influence);
    const initialB = fillrandom(side * side, 'B', random_influence);

    let N2 = Math.floor(side / 2);
    for (let i = N2 - 1; i < N2 + 1; i++) {
        let row = i * side;
        for (let j = N2 - 1; j < N2 + 1; j++) {
            let index = row + j;
            initialA[index] = 0.5;
            initialB[index] = 0.25;
        }
    }

    const fArray = filler(side, 0.01, 0.02);
    const kArray = filler(side, 0.03, 0.04);
    // const fArray = filler(side, 0.060, 0.060);
    // const kArray = filler(side, 0.062, 0.062);
    const DA = 0.16;
    const DB = 0.08;
    const dt = 1.0;
    const updateRD = new rdPipeline.PipelineRD(
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
    debugger;
    await context.connect();
    updateRD.ofVolume.attach_to_context(context);
    updateRD.attach_to_context(context);
    await updateRD.paint_on(canvas);
    //updateRD.run();
    return updateRD;
}