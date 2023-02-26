use ndarray::prelude::*;

// N-Dimensional Distance Function
fn euclidean_distance(point0 : &ArrayView1<f64>, point1 : &ArrayView1<f64>) -> f64 {
    let mut distance : f64 = 0.0;
    for i in 0..point0.len() {
        distance += (point0[[i]] - point1[[i]]).powf(2.0);
    }
    distance = distance.sqrt();
    distance
}

// K Nearest Neighbors
pub struct Knn {
    dataset : Array2<f64>
}

impl Knn {
    pub fn new(dataset : &ArrayView2<f64>) -> Self {
        Self { dataset: dataset.to_owned() }
    }

    pub fn n_neighbors(&self, k : usize, point : &ArrayView1<f64>) -> (Vec<(usize, f64)>, Array2<f64>) {
        let mut distances : Vec<(usize, f64)> = vec![];
        for i in 0..self.dataset.len() {
            let dataset_point = self.dataset.slice(s![i,..]);
            let distance = euclidean_distance(&point.view(), &dataset_point);
            distances.push((i, distance));
        }
        distances.sort_by(|a, b| { a.1.partial_cmp(&b.1).unwrap() });
        distances = distances[0..k].to_vec();

        let mut k_neighbors : Array2<f64> = Array2::zeros((k, point.len()));
        for i in 0..k {
            let index = distances[i].0;
            for j in 0..point.len() {
                k_neighbors[[i, j]] = self.dataset[[index, j]];
            }
        }

        (distances, k_neighbors)
    }
}

// KNN classifier using mode
pub struct KnnClassifier {
    knn : Knn,
    classes : Array1<f64>
}

impl KnnClassifier {
    pub fn new(features : &ArrayView2<f64>, classes : &ArrayView1<f64>) -> Self {
        Self { knn: Knn::new(features), classes: classes.to_owned() }
    }

    pub fn classify(&self, k : usize, point : &ArrayView1<f64>) -> f64 {
        let (distances, _neighbors) = self.knn.n_neighbors(k, point);
        let mut class_count : Vec<(f64, usize)> = vec![];
        'outer: for i in 0..k {
            let class = self.classes[[distances[i].0]]; // get class of index
            for j in 0..class_count.len() {
                if class_count[j].0 == class {
                    class_count[j].1 += 1;
                    continue 'outer;
                }
            }

            class_count.push((class, 1));
        }

        class_count.sort_by(|a, b| { b.1.partial_cmp(&a.1).unwrap() });
        class_count[0].0
    }
}
