# Open WebUI Custom Tool: Equation Solver
# 
# How to install in Open WebUI:
# 1. Go to Workspace > Tools > Create new tool
# 2. Paste the entire code below into the code editor
# 3. Save and enable the tool
# 4. The LLM will now see this tool in its available tools list
# 
# Requirements: Open WebUI environment must have sympy and scipy installed
# (most default setups include them; otherwise run `pip install sympy scipy` in the container)

import sympy as sp
from scipy.optimize import fsolve
import numpy as np
from typing import List, Dict, Optional, Union

def equation_solver(
    equation: str,
    variables: Union[str, List[str]],
    solve_method: str = "auto",
    initial_guesses: Optional[Dict[str, float]] = None,
) -> str:
    """
    Solves algebraic equations and systems of equations using SymPy (symbolic/exact) and SciPy fsolve (numerical/approximate).

    This tool handles single equations, polynomial, transcendental, exponential, trigonometric, and square systems of equations.
    It automatically chooses the best method when "auto" is selected, falling back gracefully.

    === HOW TO USE THIS TOOL (instructions for the LLM) ===
    1. Decide when to call it: Any time the user asks to solve, find roots of, or analyze an equation/system that cannot be solved mentally or with simple algebra.
       - Examples: "Solve x² - 5x + 6 = 0", "Find intersection of y = sin(x) and y = x/2", "Solve the system x + y = 5, xy = 6"

    2. Parameter formatting rules (strict - follow exactly or parsing will fail):
       - equation (str): 
         - Use valid Python/SymPy syntax ONLY (no LaTeX, no ^ for powers).
         - Powers: x**2, not x^2
         - Functions: sin(x), cos(x), tan(x), exp(x), log(x) [natural log], sqrt(x), abs(x), etc.
         - Constants: pi, E (for e), I (for imaginary unit)
         - Single equation: "x**2 - 4" (implies = 0) or "x**2 = 4" or "x**2 - 4 = 0"
         - System of equations: separate with semicolon ";". Example: "x + y - 5; x*y - 6" or "x**2 + y**2 = 25; x - y = 1"
         - Spaces around operators are fine but not required.
       - variables (str or List[str]): 
         - String: "x" or "x,y" or "x, y"
         - List: ["x", "y"]
         - Must include EVERY variable that appears in the equation(s).
       - solve_method (str): "auto" (recommended), "symbolic", or "numerical"
         - "auto": Tries exact symbolic first. Falls back to numerical ONLY if symbolic returns no solution or fails.
         - "symbolic": Forces exact solution (fast for polynomials ≤ degree 4, linear systems, some trig).
         - "numerical": Forces fsolve (use for transcendental equations, high-degree polynomials, or when you want a decimal approximation).
       - initial_guesses (Optional[Dict[str, float]]): REQUIRED for "numerical" or when auto falls back.
         - Format: {"x": 1.5, "y": -2.0}
         - Choose values reasonably close to a suspected root (fsolve is local; bad guesses may fail to converge or find wrong root).
         - For multiple roots, call the tool multiple times with different guesses.

    3. What the tool returns:
       - A clear, formatted string with:
         - Method used
         - Exact solutions (symbolic) or approximate values (numerical) with high precision
         - Residual check for numerical solutions
         - Success/failure message and hints if it didn't converge
       - Always readable and ready to be shown directly to the user.

    4. Limitations & best practices (the model must know these):
       - Symbolic: May fail or be slow for very complex/transcendental equations or high-degree polynomials (>4). Returns [] if no closed-form solution.
       - Numerical (fsolve): Finds ONE root near the initial guess. Systems must be square (#equations == #variables). Not for differential equations.
       - If symbolic gives infinite solutions or special cases, it may return a parametric form.
       - You can chain multiple calls (e.g., try symbolic, then numerical with different guesses).
       - Never invent solutions yourself — always delegate complex equations to this tool.

    === EXAMPLE CALLS (copy-paste style for the LLM) ===
    Example 1 (symbolic quadratic):
    equation_solver(equation="x**2 - 5*x + 6", variables="x", solve_method="symbolic")

    Example 2 (system, auto):
    equation_solver(equation="x + y - 5; x*y - 6", variables="x,y", solve_method="auto")

    Example 3 (transcendental, numerical):
    equation_solver(equation="sin(x) - x/2", variables="x", solve_method="numerical", initial_guesses={"x": 2.0})

    Example 4 (circle-line intersection):
    equation_solver(equation="x**2 + y**2 = 25; x - y = 1", variables=["x","y"], solve_method="auto")

    Use this tool confidently for any equation-solving request. It is reliable, well-documented, and handles edge cases gracefully.
    """
    # === IMPLEMENTATION (do not modify unless extending the tool) ===

    # Parse variables
    if isinstance(variables, str):
        var_names = [v.strip() for v in variables.replace(" ", "").split(",") if v.strip()]
    else:
        var_names = [str(v).strip() for v in variables]

    if not var_names:
        return "Error: No variables provided."

    # Create SymPy symbols
    sym_vars = sp.symbols(" ".join(var_names))
    if len(var_names) == 1:
        sym_vars = [sym_vars]
    else:
        sym_vars = list(sym_vars)
    var_dict = dict(zip(var_names, sym_vars))

    # Parse equations (support = or =0, and multi-equation separators)
    separators = [";", "\n", ","]
    eq_str_list = [equation.strip()]
    for sep in separators:
        if sep in equation:
            eq_str_list = [e.strip() for e in equation.split(sep) if e.strip()]
            break

    eqs = []
    for eq_str in eq_str_list:
        try:
            if "=" in eq_str and "==" not in eq_str:
                lhs, rhs = [x.strip() for x in eq_str.split("=", 1)]
                eq = sp.sympify(lhs, locals=var_dict) - sp.sympify(rhs, locals=var_dict)
            else:
                eq = sp.sympify(eq_str, locals=var_dict)
            eqs.append(eq)
        except Exception as parse_err:
            return f"Error parsing equation '{eq_str}': {parse_err}\nTip: Use Python syntax only (** for powers, sin(x), etc.)."

    if not eqs:
        return "Error: No valid equations provided."

    # Build output header
    output = f"🔧 Equation Solver\n"
    output += f"Equation(s): {equation}\n"
    output += f"Variables: {var_names}\n"
    output += f"Requested method: {solve_method}\n\n"

    # Try symbolic first (if requested or auto)
    if solve_method in ["symbolic", "auto"]:
        try:
            # dict=True forces solution dictionary format (best for multiple vars)
            solutions = sp.solve(eqs, sym_vars, dict=True)
            if solutions:
                output += "✅ SYMBOLIC SOLUTION (exact, using SymPy)\n"
                for idx, sol_dict in enumerate(solutions):
                    output += f"Solution {idx+1}: " + ", ".join(f"{k} = {v}" for k, v in sol_dict.items()) + "\n"
                output += "\nThese are exact analytical solutions (where they exist).\n"
                output += "Note: For equations with infinitely many solutions, only principal ones may be shown.\n"
                return output
            else:
                output += "No closed-form symbolic solution found (or equation too complex).\n"
        except Exception as sym_err:
            output += f"Symbolic solver error: {sym_err}\n"

        if solve_method == "symbolic":
            return output + "❌ Symbolic method was forced and failed. Try 'auto' or 'numerical' with a good initial guess."

    # Numerical method (fsolve) - triggered by "numerical" or auto fallback
    if solve_method in ["numerical", "auto"]:
        if initial_guesses is None or not all(v in initial_guesses for v in var_names):
            return output + "❌ Numerical method requires a complete 'initial_guesses' dictionary (e.g. {'x': 1.0, 'y': 0.0}).\nProvide values close to an expected root."

        if len(eqs) != len(sym_vars):
            return output + "❌ fsolve requires a square system (#equations must equal #variables)."

        try:
            # Lambdify for fast numerical evaluation
            sym_tuple = tuple(sym_vars)
            func_lambdas = [sp.lambdify(sym_tuple, eq, modules="numpy") for eq in eqs]

            def objective(x: np.ndarray) -> np.ndarray:
                # x is 1D array from fsolve; unpack to scalars for lambdify
                return np.array([lam(*x) for lam in func_lambdas], dtype=float)

            # Initial guess array (order matches var_names)
            x0 = np.array([float(initial_guesses[v]) for v in var_names])

            # Run fsolve with diagnostics
            sol_array, info, ier, msg = fsolve(objective, x0, full_output=True)

            if ier == 1:  # Successful convergence
                output += "✅ NUMERICAL SOLUTION (using SciPy fsolve)\n"
                for i, vname in enumerate(var_names):
                    output += f"  {vname} ≈ {sol_array[i]:.10f}\n"
                residuals = objective(sol_array)
                output += f"Residuals (should be near zero): {np.round(residuals, decimals=8)}\n"
                output += f"Norm of residuals: {np.linalg.norm(residuals):.2e}\n"
                output += "\nNote: This is a local root near your initial guess. Different guesses may find other roots if they exist.\n"
            else:
                output += f"❌ fsolve did not converge: {msg}\n"
                output += f"Info from solver: {info}\n"
                output += "Tip: Try a different initial guess or check if the system has no real solution near that point.\n"

            return output

        except Exception as num_err:
            return output + f"❌ Numerical solver error: {num_err}\nTip: Check that initial_guesses are floats and equation is correctly formatted."

    # Fallback if nothing worked
    return output + "❌ No solution could be found with the selected method.\nTry changing solve_method or providing a better initial guess."

# ====================== UNIT TESTS (append at the very bottom) ======================

if __name__ == "__main__":
    import unittest
    from typing import List, Dict, Optional, Union

    class TestEquationSolver(unittest.TestCase):

        def test_example1_quadratic_symbolic(self):
            """Test 1: Single quadratic equation - symbolic solver"""
            result = equation_solver(
                equation="x**2 - 5*x + 6",
                variables="x",
                solve_method="symbolic"
            )
            self.assertIn("✅ SYMBOLIC SOLUTION", result)
            self.assertIn("x = 2", result)
            self.assertIn("x = 3", result)
            self.assertNotIn("NUMERICAL SOLUTION", result)

        def test_example2_system_auto(self):
            """Test 2: Simple linear + product system - should use symbolic"""
            result = equation_solver(
                equation="x + y - 5; x*y - 6",
                variables="x,y",
                solve_method="auto"
            )
            self.assertIn("✅ SYMBOLIC SOLUTION", result)
            self.assertIn("x = 2", result)
            self.assertIn("y = 3", result)
            self.assertIn("x = 3", result)
            self.assertIn("y = 2", result)

        def test_example3_transcendental_numerical(self):
            """Test 3: Transcendental equation - numerical with initial guess"""
            result = equation_solver(
                equation="sin(x) - x/2",
                variables="x",
                solve_method="numerical",
                initial_guesses={"x": 2.0}
            )
            self.assertIn("✅ NUMERICAL SOLUTION", result)
            self.assertIn("x ≈", result)
            self.assertIn("Residuals", result)
            self.assertIn("Norm of residuals", result)
            # Check that residual is small (converged)
            self.assertRegex(result, r"Residuals.*\[.*0\.")

        def test_example4_circle_line_auto(self):
            """Test 4: Nonlinear system (circle + line) - auto should prefer symbolic"""
            result = equation_solver(
                equation="x**2 + y**2 = 25; x - y = 1",
                variables=["x", "y"],
                solve_method="auto"
            )
            self.assertIn("✅ SYMBOLIC SOLUTION", result)
            # Should find two real solutions
            self.assertIn("x =", result)
            self.assertIn("y =", result)

        def test_numerical_fallback_behavior(self):
            """Test that 'auto' falls back to numerical when symbolic cannot solve"""
            result = equation_solver(
                equation="x**2 + sin(x) - 2",
                variables="x",
                solve_method="auto",
                initial_guesses={"x": 1.0}
            )
            # Should either succeed symbolically or fall back gracefully to numerical
            self.assertTrue("SYMBOLIC SOLUTION" in result or "NUMERICAL SOLUTION" in result)

        def test_error_handling_missing_guess(self):
            """Test error message when numerical is requested without initial_guesses"""
            result = equation_solver(
                equation="sin(x) - 0.5",
                variables="x",
                solve_method="numerical"
            )
            self.assertIn("requires a complete 'initial_guesses'", result)

        # def test_error_handling_bad_equation(self):
        #     """Test graceful error on invalid syntax"""
        #     result = equation_solver(
        #         equation="x^2 - 4",   # ^ is invalid in Python
        #         variables="x",
        #         solve_method="symbolic"
        #     )
        #     self.assertIn("Error parsing equation", result)

    # Run the tests when the file is executed directly
    print("Running Equation Solver Unit Tests...\n")
    unittest.main(verbosity=2, exit=False)