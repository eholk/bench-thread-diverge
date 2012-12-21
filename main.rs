extern mod std;
use std::time::precise_time_s;

extern mod OpenCL;

use OpenCL::hl::*;
use OpenCL::vector::*;

const N: uint = 1024;

fn main() {
    let ctx = create_compute_context();

    let r = rand::Rng();

    let A = vec::from_fn(N * N, |_| r.gen_f64());
    let A = Vector::from_vec(ctx, A);

    let B = vec::from_fn(N * N, |_| r.gen_f64());
    let B = Vector::from_vec(ctx, B);

    let C = vec::from_elem(N * N, 0f64);
    let C = Vector::from_vec(ctx, C);

    let result = io::read_whole_file_str(&path::Path("kernels.cl"));
    let source = match move result {
        Ok(move src) => src,
        Err(move e) => fail fmt!("Could not read file: %?", e)
    };

    let program = ctx.create_program_from_source(source);
    match program.build(ctx.device) {
        Ok(()) => (),
        Err(build_log) => {
            io::println("Error building OpenCL program!");
            io::println("");
            io::println(build_log);
            fail
        }
    }

    let k = program.create_kernel("MyAdd");

    let time = time_kernel(ctx, &k, &A, &B, &C);

    io::println(fmt!("MyAdd:\t%? msec/kernel", time));
}

fn time_kernel(ctx: @ComputeContext, k: &Kernel,
               A: &Vector<f64>,
               B: &Vector<f64>,
               C: &Vector<f64>) -> float
{
    const REP: uint = 500;

    const LOCAL_SIZE: int = 256;

    k.set_arg(0, A);
    k.set_arg(1, B);
    k.set_arg(2, C);
    k.set_arg(3, &N);

    // Call the timer once to warm up.
    precise_time_s();

    // Do it once to avoid timing inconsistencies.
    enqueue_nd_range_kernel(
        &ctx.q,
        k,
        1, 0, N as int, LOCAL_SIZE);

    // Loop one
    let start1 = precise_time_s();
    for REP.times || {
        enqueue_nd_range_kernel(
            &ctx.q,
            k,
            1, 0, N as int, LOCAL_SIZE);
    }
    let stop1 = precise_time_s();

    let elapsed1 = stop1 - start1;

    // Loop two
    let start2 = precise_time_s();
    for REP.times || {
        enqueue_nd_range_kernel(
            &ctx.q,
            k,
            1, 0, N as int, LOCAL_SIZE);
        enqueue_nd_range_kernel(
            &ctx.q,
            k,
            1, 0, N as int, LOCAL_SIZE);
    }
    let stop2 = precise_time_s();

    let elapsed2 = stop2 - start2;

    (elapsed2 - elapsed1) / (REP as float) * 1000000f
}
