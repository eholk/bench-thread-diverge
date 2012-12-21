extern mod OpenCL;

use OpenCL::hl::*;
use OpenCL::vector::*;

const N: uint = 1200;

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
    
    k.set_arg(0, &A);
    k.set_arg(1, &B);
    k.set_arg(2, &C);
    k.set_arg(3, &N);

    enqueue_nd_range_kernel(
        &ctx.q,
        &k,
        1, 0, N as int, 1);
}
