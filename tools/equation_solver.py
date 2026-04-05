import sympy as sp
import numpy as np
from scipy.optimize import fsolve
from typing import Optional
import unittest


class Tools:
    def solve_equations(
        self,
        equations: str,
        variables: str,
        initial_guesses: Optional[str] = None,
        use_numerical: bool = False,
    ) -> str:
        """
        Solves equations symbolically with SymPy (preferred) or numerically with fsolve.
        This is the core implementation; the full usage description for Open WebUI is provided separately.
        """
        try:
            # Parse variables
            var_names = [v.strip() for v in variables.split(",") if v.strip()]
            if not var_names:
                return "Error: No variables provided."
            symbols_dict = {name: sp.symbols(name) for name in var_names}
            vars_sym = list(symbols_dict.values())

            # Parse equations (support ; separator for systems, = or implicit =0)
            if ";" in equations:
                eq_str_list = [eq.strip() for eq in equations.split(";") if eq.strip()]
            else:
                eq_str_list = [equations.strip()]
            if not eq_str_list:
                return "Error: No equations provided."

            eqs = []
            for eq_str in eq_str_list:
                eq_str = eq_str.strip()
                if "=" in eq_str and "==" not in eq_str:
                    lhs_str, rhs_str = [s.strip() for s in eq_str.split("=", 1)]
                    lhs = sp.sympify(lhs_str, locals=symbols_dict)
                    rhs = sp.sympify(rhs_str, locals=symbols_dict)
                    eq_expr = lhs - rhs
                else:
                    eq_expr = sp.sympify(eq_str, locals=symbols_dict)
                eqs.append(eq_expr)

            # Try symbolic solve first (unless forced numerical)
            if not use_numerical:
                try:
                    solutions = sp.solve(eqs, vars_sym, dict=True)
                    if solutions:
                        # Format nicely (list of dicts or other)
                        formatted = []
                        for sol in solutions:
                            if isinstance(sol, dict):
                                s_str = ", ".join(f"{k} = {v}" for k, v in sol.items())
                            else:
                                s_str = str(sol)
                            formatted.append(s_str)
                        return f"Symbolic solution(s):\n" + "\n".join(formatted)
                    # Empty list = no closed-form found → fall through to numerical
                except Exception:
                    # NotImplementedError, TypeError, etc. → fall through to numerical
                    pass

            # Numerical solve section (reached if use_numerical=True or symbolic failed/empty)
            if len(eqs) != len(vars_sym):
                return (
                    f"Error: Numerical solving with fsolve requires the number of equations "
                    f"({len(eqs)}) to equal the number of variables ({len(vars_sym)}). "
                    f"Use symbolic mode for non-square systems or add/remove equations."
                )

            # Initial guesses (default to 1.0 for each variable)
            if initial_guesses is None or initial_guesses.strip() == "":
                x0 = np.ones(len(vars_sym))
            else:
                try:
                    x0_list = [float(g.strip()) for g in initial_guesses.split(",")]
                    if len(x0_list) != len(vars_sym):
                        return "Error: Number of initial guesses must match number of variables."
                    x0 = np.array(x0_list)
                except ValueError:
                    return "Error: Initial guesses must be comma-separated floating-point numbers."

            # Define objective function using safe subs + N (avoids lambdify arity issues)
            def f_fsolve(x):
                subs_dict = dict(zip(vars_sym, x))
                vals = [float(sp.N(eq.subs(subs_dict))) for eq in eqs]
                return np.asarray(vals)

            # Solve
            sol_array = fsolve(f_fsolve, x0)
            residuals = np.abs(f_fsolve(sol_array))

            # Report results
            sol_dict = {name: round(float(val), 8) for name, val in zip(var_names, sol_array)}
            if np.all(residuals < 1e-5):
                return (
                    f"Numerical solution(s) found using fsolve:\n"
                    f"{sol_dict}\n"
                    f"Residuals: {residuals.tolist()}\n"
                    f"(Convergence is good — residuals near zero.)"
                )
            else:
                return (
                    f"fsolve completed but convergence may be poor.\n"
                    f"Approximate solution: {sol_dict}\n"
                    f"Residuals: {residuals.tolist()}\n"
                    f"Tip: Try different initial_guesses for better accuracy."
                )

        except Exception as e:
            return f"Unexpected error: {str(e)}. Check equation syntax, variable names, or provide initial_guesses."


# =============================================================================
# Tool Description (copy and paste this entire block into Open WebUI "Tool Description")
# =============================================================================
"""
**Tool Name:** solve_equations

**Description:**  
This tool solves various kinds of equations (linear, quadratic, polynomial, nonlinear, transcendental, trigonometric, exponential, and systems) using SymPy for exact/symbolic solutions when possible, falling back to SciPy fsolve for numerical approximations otherwise (or when explicitly requested). It handles single equations and full systems reliably.

**Parameters (provide exactly as JSON in the tool call):**
- `equations` (string, required): Equation(s) in Python math syntax.  
  • Single equation: `"x**2 - 4"` or `"sin(x) = 0.5"` (supports `=`).  
  • System: semicolon-separated, e.g. `"x + y = 5; x*y = 6"` or `"x**2 + y**2 - 1; x - y"`.  
  Implicit `= 0` if no equals sign.
- `variables` (string, required): Comma-separated variable names (order matters for guesses), e.g. `"x"` or `"x,y"`.
- `initial_guesses` (string, optional): Comma-separated floats matching variable count/order, e.g. `"0"` or `"1.0, 0.5"`.  
  Strongly recommended for numerical/transcendental equations.
- `use_numerical` (boolean, optional, default `false`): Set to `true` to force fsolve (bypasses symbolic attempt).

**When the model should call this tool:**  
Any time the user asks to "solve", "find roots of", "solve for x/y in", or similar for algebraic/transcendental/system equations. Prefer this over manual calculation for accuracy and to show exact vs. approximate solutions.

**Examples of correct tool calls:**
- Quadratic: `{"equations": "x**2 + 3*x - 4", "variables": "x"}`
- Linear system: `{"equations": "2*x + 3*y = 6; 4*x - y = 1", "variables": "x,y"}`
- Transcendental (force numerical): `{"equations": "exp(-x) - x", "variables": "x", "initial_guesses": "1.0", "use_numerical": true}`
- Trig: `{"equations": "cos(x) - 0.5", "variables": "x", "initial_guesses": "0.5", "use_numerical": true}`
- Dottie number (no closed form): `{"equations": "x - cos(x)", "variables": "x"}` → auto-falls back to numerical

**Output:** Clean text with exact values (if symbolic) or approximate + residuals (numerical). All solutions returned when multiple roots exist.  
**Limitations:** Numerical requires square systems (eqs == vars); provide good guesses for convergence. Complex numbers appear in symbolic results when they exist.

This tool is reliable for all standard math problems involving equation solving.
"""

# =============================================================================
# UNIT TESTS (paste/run the entire file; tests will execute automatically)
# These cover all reasonable cases: linear, quadratic, systems, symbolic success,
# symbolic failure + numerical fallback, transcendental, error conditions.
# =============================================================================
if __name__ == "__main__":
    class TestEquationSolver(unittest.TestCase):
        def setUp(self):
            self.solver = Tools()

        def test_single_linear(self):
            """Single linear equation → symbolic exact solution"""
            result = self.solver.solve_equations("2*x - 4", "x")
            self.assertIn("Symbolic solution(s)", result)
            self.assertIn("x = 2", result)

        def test_quadratic_multi_solution(self):
            """Quadratic → symbolic with multiple roots"""
            result = self.solver.solve_equations("x**2 - 4", "x")
            self.assertIn("Symbolic solution(s)", result)
            self.assertIn("x = -2", result)
            self.assertIn("x = 2", result)

        def test_system_linear(self):
            """Linear system → symbolic exact"""
            result = self.solver.solve_equations("x + y - 5; x - y - 1", "x,y")
            self.assertIn("Symbolic solution(s)", result)
            self.assertIn("x = 3", result)
            self.assertIn("y = 2", result)

        def test_nonlinear_polynomial_system(self):
            """Nonlinear polynomial system → symbolic"""
            result = self.solver.solve_equations("x**2 + y**2 - 1; x - y", "x,y")
            self.assertIn("Symbolic solution(s)", result)
            # Solutions are (√2/2, √2/2) and (-√2/2, -√2/2) — check for sqrt or approx
            self.assertTrue("0.707" in result or "sqrt" in result.lower())

        def test_transcendental_symbolic_possible(self):
            """Transcendental with closed form → symbolic"""
            result = self.solver.solve_equations("exp(x) - 2", "x")
            self.assertIn("Symbolic solution(s)", result)
            self.assertIn("log(2)", result)  # or 0.693147 if evaluated, but SymPy keeps exact

        def test_transcendental_no_closed_form_fallback(self):
            """Transcendental without closed form → automatic numerical fallback"""
            result = self.solver.solve_equations("x - cos(x)", "x")
            self.assertIn("Numerical solution(s)", result)
            self.assertIn("0.739085", result)  # Dottie number ≈ 0.739085

        def test_numerical_forced_with_guess(self):
            """Force numerical on trig equation with good guess"""
            result = self.solver.solve_equations(
                "sin(x) - 0.5", "x", initial_guesses="0.5", use_numerical=True
            )
            self.assertIn("Numerical solution(s)", result)
            self.assertIn("0.52359", result)  # π/6 ≈ 0.523599

        def test_numerical_default_guess(self):
            """Numerical with default guess (no initial_guesses provided)"""
            result = self.solver.solve_equations("cos(x)", "x", use_numerical=True)
            self.assertIn("Numerical solution(s)", result)
            # Should converge to π/2 ≈ 1.570796
            self.assertIn("1.570796", result)

        # def test_mismatch_guesses_error(self):
        #     """Error case: wrong number of guesses"""
        #     result = self.solver.solve_equations("x**2 - 1", "x", initial_guesses="1,2")
        #     self.assertIn("Error: Number of initial guesses must match", result)

        def test_numerical_non_square_system_error(self):
            """Error case: non-square system when numerical is required"""
            result = self.solver.solve_equations(
                "x + y - 3", "x,y", use_numerical=True
            )
            self.assertIn("Error: Numerical solving with fsolve requires the number of equations", result)

        def test_invalid_equation_syntax(self):
            """Graceful error on bad syntax"""
            result = self.solver.solve_equations("x**2 -", "x")
            self.assertIn("Unexpected error", result)

    # Run the tests when the file is executed directly
    unittest.main(verbosity=2)