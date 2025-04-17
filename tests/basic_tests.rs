use xla::{ArrayElement, Result};

#[test]
fn add_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    let mat = [
        [1.0f32, 2.0f32, 3.0f32].as_slice(),
        [4.0f32, 5.0f32, 6.0f32].as_slice(),
        [7.0f32, 8.0f32, 9.0f32].as_slice(),
    ];
    let ones = [1.0f32, 1.0f32, 1.0f32];

    // test matrix adding and multiplying
    let builder = xla::XlaBuilder::new("test");
    let cst42 = builder.constant_r0(42f32)?;
    let vec = builder.constant_r1(&ones)?;
    let cstmat = builder.constant_r2(&mat)?;
    let sum = (&cst42 + &cstmat)?;
    let sum = sum.matmul(&vec)?;
    let computation = sum.build()?;
    let result_mat = client.compile(&computation)?;
    let result_mat = result_mat.execute::<xla::Literal>(&[])?;
    let result_mat = result_mat[0][0].to_literal_sync()?;
    assert_eq!(result_mat.element_count(), 3);
    assert_eq!(result_mat.array_shape()?, xla::ArrayShape::new::<f32>(vec![3]));
    assert_eq!(result_mat.to_vec::<f32>()?, [132., 141., 150.]);

    // same test but with matrix transposed
    let builder = xla::XlaBuilder::new("test");
    let cst42 = builder.constant_r0(42f32)?;
    let vec = builder.constant_r1(&ones)?;
    let cstmat = builder.constant_r2(&mat)?;
    let cstmat = cstmat.transpose(&[1, 0])?;
    let sum = (&cst42 + &cstmat)?;
    let sum = sum.matmul(&vec)?;
    let computation = sum.build()?;
    let result_mat = client.compile(&computation)?;
    let result_mat = result_mat.execute::<xla::Literal>(&[])?;
    let result_mat = result_mat[0][0].to_literal_sync()?;
    assert_eq!(result_mat.element_count(), 3);
    assert_eq!(result_mat.array_shape()?, xla::ArrayShape::new::<f32>(vec![3]));
    assert_eq!(result_mat.to_vec::<f32>()?, [138., 141., 144.]);

    let builder = xla::XlaBuilder::new("test");
    let cst42 = builder.constant_r0(42f32)?;
    let cst43 = builder.constant_r1c(43f32, 2)?;
    let sum = (&cst42 + &cst43)?;
    let computation = sum.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<xla::Literal>(&[])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.element_count(), 2);
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![2]));
    assert_eq!(result.get_first_element::<f32>()?, 85.);
    assert_eq!(result.to_vec::<f32>()?, [85., 85.]);
    Ok(())
}

#[test]
fn malformed_mat() -> Result<()> {
    let bad_mat = [
        [1.0f32, 2.0f32, 3.0f32].as_slice(),
        [4.0f32, 5.0f32, 6.0f32].as_slice(),
        [7.0f32, 8.0f32].as_slice(),
    ];

    let result = std::panic::catch_unwind(|| {
        let builder = xla::XlaBuilder::new("test");
        builder.constant_r2(&bad_mat)
    });
    assert!(result.is_err_and(|e| {
        let msg = match e.downcast_ref::<&'static str>() {
            Some(s) => *s,
            None => match e.downcast_ref::<String>() {
                Some(s) => &s[..],
                None => "Box<dyn Any>",
            },
        };
        msg == "all rows must have the same number of columns!"
    }));

    let good_mat = [
        [1.0f32, 2.0f32, 3.0f32].as_slice(),
        [4.0f32, 5.0f32, 6.0f32].as_slice(),
        [7.0f32, 8.0f32, 9.0f32].as_slice(),
    ];
    let result = std::panic::catch_unwind(|| {
        let builder = xla::XlaBuilder::new("test");
        builder.constant_r2(&good_mat)
    });
    assert!(result.is_ok());

    Ok(())
}

#[test]
fn triangular_solve() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let a_ = [
        [1.0f32, 2.0f32, 3.0f32].as_slice(),
        [4.0f32, 5.0f32, 6.0f32].as_slice(),
        [7.0f32, 8.0f32, 9.0f32].as_slice(),
    ];
    let a = builder.constant_r2(&a_)?;

    // let b_ = [[2.0f32].as_slice(), [3.0f32].as_slice(), [4.0f32].as_slice()];
    let b = builder.constant_r2(&a_)?;

    let t_solve = a.triangular_solve(&b, false, true, false, 1)?;
    let computation = t_solve.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<xla::Literal>(&[])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.element_count(), 9);
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![3, 3]));

    Ok(())
}

#[test]
fn sum_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[2], "x")?;
    let sum = x.reduce_sum(&[], false)?.build()?.compile(&client)?;
    let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = sum.execute::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [4.2, 1.337]);

    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[-2], "x")?;
    let sum = x.reduce_sum(&[0], false)?.build()?.compile(&client)?;
    let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = sum.execute::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [5.5369997]);
    // Dimensions got reduced.
    // assert_eq!(result.array_shape()?.dims(), []);

    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[-2], "x")?;
    let sum = x.reduce_sum(&[0], true)?.build()?.compile(&client)?;
    let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = sum.execute::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [5.5369997]);
    // keep_dims = true in this case.
    assert_eq!(result.array_shape()?.dims(), [1]);
    Ok(())
}

#[test]
fn mean_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[-2], "x")?;
    let sum = x.reduce_mean(&[0], false)?.build()?.compile(&client)?;
    let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = sum.execute::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [2.7684999]);
    // Dimensions got reduced.
    // assert_eq!(result.array_shape()?.dims(), []);
    Ok(())
}

#[test]
fn tuple_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[-1], "x")?;
    let y = builder.parameter(1, f32::TY, &[2], "x")?;
    let tuple = builder.tuple(&[x, y])?.build()?.compile(&client)?;
    let x = xla::Literal::scalar(3.1f32);
    let y = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = tuple.execute::<xla::Literal>(&[x, y])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.shape()?.tuple_size(), Some(2));
    let mut result = result;
    let result = result.decompose_tuple()?;
    assert_eq!(result[1].to_vec::<f32>()?, [4.2, 1.337]);
    assert_eq!(result[0].to_vec::<f32>()?, [3.1]);
    Ok(())
}

#[test]
fn tuple_literal() -> Result<()> {
    let x = xla::Literal::scalar(3.1f32);
    let y = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = xla::Literal::tuple(vec![x, y]);
    assert_eq!(result.shape()?.tuple_size(), Some(2));
    let mut result = result;
    let result = result.decompose_tuple()?;
    assert_eq!(result[1].to_vec::<f32>()?, [4.2, 1.337]);
    assert_eq!(result[0].to_vec::<f32>()?, [3.1]);
    Ok(())
}

#[test]
fn get_hlo_computations() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[-2], "x")?;
    let mean = x.reduce_mean(&[0], false)?;
    let computation = mean.build()?;

    let proto = computation.proto();
    println!("module proto {:?}", proto);
    let comps = proto.computations()?;
    println!("computation protos {:?}", comps);
    for comp in comps {
        let instrs = comp.instructions()?;
        println!("instructions for comp {:?} {:?}", comp, instrs);

        for instr in instrs {
            println!("opcode: {:?}", instr.opcode()?)
        }
    }
    assert!(true == true);

    let exec = computation.compile(&client)?;
    // let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    // let result = exec.execute::<xla::Literal>(&[input])?;
    // let result = result[0][0].to_literal_sync()?;
    // assert_eq!(result.to_vec::<f32>()?, [2.7684999]);
    // Dimensions got reduced.
    // assert_eq!(result.array_shape()?.dims(), []);
    Ok(())
}
