use burn::prelude::*;
use burn::backend::Wgpu;

type Backend = Wgpu;

fn main() {
    let device = Default::default();
    // create two tensors,
    // "a": from an array
    // "b": full of ones
    let tensor_a = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_b = Tensor::<Backend, 2>::ones_like(&tensor_a);

    println!("[INFO] result: {}", tensor_a + tensor_b);
}
