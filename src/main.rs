mod knn;

use knn::KnnClassifier;
use ndarray::prelude::*;

fn separator() {
    println!();
    println!("====================");
    println!();
}

fn main() {
    println!("[*] Machine Learning from Scratch");

    separator();

    println!("[*] 1. K Nearest Neighbors Classifier");

    // KNN Classifier Training/Testing Data
    let chocolate_data : Array2<f64> = array![
        // body fat %, eats chocolate
        [35.0, 1.0],
        [19.2, 1.0],
        [14.1, 0.0],
        [12.6, 0.0],
        [11.4, 0.0],
        [15.5, 1.0],
        [15.2, 0.0],
        [16.7, 1.0],
        [19.3, 1.0],
        [6.9,  0.0]
    ];

    let chocolate_test_data : Array2<f64> = array![
        // body fat %, eats chocolate
	    [12.9, 0.0],
	    [36.2, 1.0],
	    [11.7, 0.0],
	    [15.4, 0.0], // odd one out (bad data)
	    [15.7, 1.0],
	    [11.2, 0.0],
	    [10.7, 0.0],
	    [6.2,  0.0],
	    [53.4, 1.0],
	    [34.2, 1.0]
	];

    let mut chocolate_features : Array2<f64> = Array2::zeros((10, 1));
    for i in 0..chocolate_data.nrows() {
        chocolate_features[[i, 0]] = chocolate_data[[i, 0]];
    }

    let mut chocolate_test_features : Array2<f64> = Array2::zeros((10, 1));
    for i in 0..chocolate_test_data.nrows() {
        chocolate_test_features[[i, 0]] = chocolate_test_data[[i, 0]];
    }

    let chocolate_labels = chocolate_data
        .t()
        .slice(s![1, ..])
        .to_owned();

    let chocolate_test_labels = chocolate_test_data
        .t()
        .slice(s![1, ..])
        .to_owned();

    // Show Data
    println!();
    println!("[*] Training Data:");
    println!("{}", chocolate_data);

    println!();
    println!("[*] Training Data:");
    println!("{}", chocolate_data);
    println!();

    let classifier = KnnClassifier::new(&chocolate_features.view(), &chocolate_labels.view());
    let total_predictions = chocolate_test_features.nrows();
    let mut correct_predictions = 0;
    for i in 0..total_predictions {
        let point = chocolate_test_features.slice(s![i,..]);
        let class = classifier.classify(3, &point.view());
        let correct_class = chocolate_test_labels[[i]];
        let mut message = format!("[*] Prediction: {} -> {}", point, class);
        if class == correct_class {
            message = format!("{} | Correct", message);
            correct_predictions += 1;
        } else {
            message = format!("{} | Expected: {}", message, correct_class);
        }
        println!("{}", message);
    }

    println!();
    println!("[*] Accuracy: {}%", (correct_predictions as f64 / total_predictions as f64) * 100.0);

    separator();
    /*
    let stock_data : Array2<f64> = array![
        // open price, close price
        [9.0, 10.0],
        [10.0, 12.5],
        [12.5, 11.8],
        [11.8, 11.2],
        [11.2, 12.4],
        [12.4, 12.2],
        [12.2, 14.1],
        [14.1, 13.8],
        [13.8, 14.4],
        [14.4, 15.0]
    ];

    let mut stock_features : Array2<f64> = Array2::zeros((10, 1));
    for i in 0..stock_data.nrows() {
        stock_features[[i, 0]] = stock_data[[i, 0]];
    }
    */
}
