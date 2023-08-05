#![allow(unused)]

use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;

fn main() {
    extern crate argmin;
    use argmin::core::{CostFunction, Error, Gradient, Hessian};

    /// First, we create a struct called `Rosenbrock` for your problem
    struct Rosenbrock {
        a: f64,
        b: f64,
    }

    /// Implement `CostFunction` for `Rosenbrock`
    ///
    /// First, we need to define the types which we will be using. Our parameter
    /// vector will be a `Vec` of `f64` values and our cost function value will
    /// be a 64 bit floating point value.
    /// This is reflected in the associated types `Param` and `Output`, respectively.
    ///
    /// The method `cost` then defines how the cost function is computed for a
    /// parameter vector `p`. Note that we have access to the fields `a` and `b`
    /// of `Rosenbrock`.
    impl CostFunction for Rosenbrock {
        /// Type of the parameter vector
        type Param = Vec<f64>;
        /// Type of the return value computed by the cost function
        type Output = f64;

        /// Apply the cost function to a parameter `p`
        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            // Evaluate 2D Rosenbrock function
            let a = self.a;
            let b = self.b;
            let x = p[0];
            let y = p[1];
            Ok((a - x).powi(2) + b * (y - x.powi(2)).powi(2))
        }
    }

    /// Implement `Gradient` for `Rosenbrock`
    ///
    /// Similarly to `CostFunction`, we need to define the type of our parameter
    /// vectors and of the gradient we are computing. Since the gradient is also
    /// a vector, it is of type `Vec<f64>` just like `Param`.
    impl Gradient for Rosenbrock {
        /// Type of the parameter vector
        type Param = Vec<f64>;
        /// Type of the gradient
        type Gradient = Vec<f64>;

        /// Compute the gradient at parameter `p`.
        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            let a = self.a;
            let b = self.b;
            let x = p[0];
            let y = p[1];
            // Compute gradient of 2D Rosenbrock function
            let dx = -2.0 * a + 4.0 * b * x.powi(3) - 4.0 * b * x * y + 2.0 * x;
            let dy = -2.0 * b * (x.powi(2) - y);
            Ok(vec![dx, dy])
        }
    }

    let cost = Rosenbrock { a: 1.0, b: 100.0 };
    let init_param: Vec<f64> = vec![-1.2, 2.0];

    // Set up line search needed by `SteepestDescent`
    let linesearch = MoreThuenteLineSearch::new();

    // Set up solver -- `SteepestDescent` requires a linesearch
    let solver = SteepestDescent::new(linesearch);

    // Create an `Executor` object
    let res = Executor::new(cost, solver)
        // Via `configure`, one has access to the internally used state.
        // This state can be initialized, for instance by providing an
        // initial parameter vector.
        // The maximum number of iterations is also set via this method.
        // In this particular case, the state exposed is of type `IterState`.
        // The documentation of `IterState` shows how this struct can be
        // manipulated.
        // Population based solvers use `PopulationState` instead of
        // `IterState`.
        .configure(|state| {
            state
                // Set initial parameters (depending on the solver,
                // this may be required)
                .param(init_param)
                // Set maximum iterations to 10
                // (optional, set to `std::u64::MAX` if not provided)
                .max_iters(10000)
                // Set target cost. The solver stops when this cost
                // function value is reached (optional)
                .target_cost(0.0)
        })
        // run the solver on the defined problem
        .run()
        .unwrap();

    // print result
    println!("{}", res);

    // Extract results from state

    // Best parameter vector
    let best = res.state().get_best_param().unwrap();

    // Cost function value associated with best parameter vector
    let best_cost = res.state().get_best_cost();

    // Check the execution status
    // let termination_status = res.state().get_termination_status();

    // Optionally, check why the optimizer terminated (if status is terminated)
    let termination_reason = res.state().get_termination_reason();

    // Time needed for optimization
    let time_needed = res.state().get_time().unwrap();

    // Total number of iterations needed
    let num_iterations = res.state().get_iter();

    // Iteration number where the last best parameter vector was found
    let num_iterations_best = res.state().get_last_best_iter();

    // Number of evaluation counts per method (Cost, Gradient)
    let function_evaluation_counts = res.state().get_func_counts();
}
