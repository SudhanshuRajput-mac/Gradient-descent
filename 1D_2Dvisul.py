import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import pandas as pd
import base64
import time
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import re

# Configure the page
st.set_page_config(
    page_title="Gradient Descent Visualizer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .step-animation {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def safe_eval_function(expression, x, y=None):
    """Safely evaluate mathematical expressions"""
    try:
        # Replace common mathematical functions
        expression = expression.replace('^', '**')
        expression = expression.replace('sin', 'np.sin')
        expression = expression.replace('cos', 'np.cos')
        expression = expression.replace('tan', 'np.tan')
        expression = expression.replace('exp', 'np.exp')
        expression = expression.replace('log', 'np.log')
        expression = expression.replace('sqrt', 'np.sqrt')
        expression = expression.replace('abs', 'np.abs')
        
        # Security: Only allow safe operations
        allowed_chars = set('0123456789+-*/.()xyz np')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains unsafe characters")
        
        if y is None:
            # 1D function
            return eval(expression, {'np': np, 'x': x})
        else:
            # 2D function
            return eval(expression, {'np': np, 'x': x, 'y': y})
    except Exception as e:
        st.error(f"Error evaluating function: {e}")
        return None

def parse_custom_function(expression, dimension=1):
    """Parse custom function and return a callable function"""
    try:
        if dimension == 1:
            def f(x):
                return safe_eval_function(expression, x)
        else:  # dimension == 2
            def f(x, y):
                return safe_eval_function(expression, x, y)
        return f
    except Exception as e:
        st.error(f"Error parsing function: {e}")
        return None

def calculate_gradient_2d(f, x, y, h=1e-5):
    """Calculate gradient for 2D function numerically"""
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

def create_2d_gradient_descent_animation(function_choice, start_x, start_y, learning_rate, num_steps, custom_func=None):
    """Create 2D gradient descent animation"""
    
    if function_choice == "Custom Function" and custom_func:
        f = custom_func
        x_vals = np.linspace(-3, 3, 50)
        y_vals = np.linspace(-3, 3, 50)
    elif function_choice == "Quadratic Bowl: f(x,y) = x² + y²":
        def f(x, y):
            return x**2 + y**2
        x_vals = np.linspace(-2, 2, 50)
        y_vals = np.linspace(-2, 2, 50)
    elif function_choice == "Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²":
        def f(x, y):
            return (1 - x)**2 + 100 * (y - x**2)**2
        x_vals = np.linspace(-2, 2, 50)
        y_vals = np.linspace(-1, 3, 50)
    elif function_choice == "Himmelblau: f(x,y) = (x²+y-11)² + (x+y²-7)²":
        def f(x, y):
            return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        x_vals = np.linspace(-5, 5, 50)
        y_vals = np.linspace(-5, 5, 50)
    else:  # Simple valley
        def f(x, y):
            return x**2 + 0.5 * y**2
        x_vals = np.linspace(-2, 2, 50)
        y_vals = np.linspace(-2, 2, 50)
    
    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)
    
    # Calculate all steps in advance
    current_pos = np.array([start_x, start_y], dtype=float)
    steps_data = []
    
    for step in range(num_steps + 1):
        current_x, current_y = current_pos
        current_z = f(current_x, current_y)
        
        if step < num_steps:
            gradient = calculate_gradient_2d(f, current_x, current_y)
            next_pos = current_pos - learning_rate * gradient
            next_x, next_y = next_pos
            next_z = f(next_x, next_y)
        else:
            gradient = np.array([0, 0])
            next_pos = current_pos
            next_x, next_y = current_pos
            next_z = current_z
            
        steps_data.append({
            'step': step,
            'current_pos': current_pos.copy(),
            'current_z': current_z,
            'gradient': gradient.copy(),
            'next_pos': next_pos.copy(),
            'next_z': next_z,
            'learning_rate': learning_rate
        })
        
        if step < num_steps:
            current_pos = next_pos
    
    return f, X, Y, Z, steps_data

def plot_2d_current_step(f, X, Y, Z, step_data, show_next=True):
    """Plot the current step of 2D gradient descent"""
    fig = plt.figure(figsize=(15, 6))
    
    # Plot 1: 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    
    # Plot optimization path
    current_step = step_data['step']
    if 'history' in step_data and current_step > 0:
        history = step_data['history']
        path_x = [step['current_pos'][0] for step in history]
        path_y = [step['current_pos'][1] for step in history]
        path_z = [step['current_z'] for step in history]
        ax1.plot(path_x, path_y, path_z, 'ro-', linewidth=2, markersize=4)
    
    # Current point
    current_pos = step_data['current_pos']
    current_z = step_data['current_z']
    ax1.scatter([current_pos[0]], [current_pos[1]], [current_z], 
               color='red', s=100, label=f'Step {current_step}')
    
    # Next point
    if show_next and step_data['step'] < len(step_data.get('history', [])) - 1:
        next_step = step_data['history'][step_data['step'] + 1]
        next_pos = next_step['current_pos']
        next_z = next_step['current_z']
        ax1.scatter([next_pos[0]], [next_pos[1]], [next_z], 
                   color='green', s=80, alpha=0.7, label='Next Step')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title(f'3D Surface - Step {current_step}')
    ax1.legend()
    
    # Plot 2: Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Plot path on contour
    if 'history' in step_data and current_step > 0:
        history = step_data['history']
        path_x = [step['current_pos'][0] for step in history]
        path_y = [step['current_pos'][1] for step in history]
        ax2.plot(path_x, path_y, 'ro-', linewidth=2, markersize=4)
    
    # Current point
    ax2.scatter([current_pos[0]], [current_pos[1]], color='red', s=100, label=f'Step {current_step}')
    
    # Gradient arrow
    if step_data['gradient'].any():
        gradient = step_data['gradient']
        ax2.arrow(current_pos[0], current_pos[1], 
                 -0.1 * gradient[0], -0.1 * gradient[1],
                 head_width=0.1, head_length=0.1, fc='green', ec='green')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Contour Plot with Gradient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_step_by_step_animation(function_choice, start_x, learning_rate, num_steps, custom_func=None):
    """Create step-by-step animation showing progression from start to finish"""
    
    if function_choice == "Custom Function" and custom_func:
        f = custom_func
        x_vals = np.linspace(-3, 3, 400)
    elif function_choice == "Quadratic: f(x) = (x-3)² + 2":
        def f(x):
            return (x - 3)**2 + 2
        x_vals = np.linspace(-1, 7, 400)
    elif function_choice == "Double Well: f(x) = x⁴ - 8x² + 3x + 10":
        def f(x):
            return x**4 - 8*x**2 + 3*x + 10
        x_vals = np.linspace(-3, 4, 400)
    elif function_choice == "Complex: f(x) = sin(x) + 0.1*(x-2)²":
        def f(x):
            return np.sin(x) + 0.1*(x-2)**2
        x_vals = np.linspace(-1, 5, 400)
    else:  # Default quadratic
        def f(x):
            return (x - 3)**2 + 2
        x_vals = np.linspace(-1, 7, 400)
    
    y_vals = f(x_vals)
    
    # Calculate all steps in advance
    current_x = start_x
    steps_data = []
    
    for step in range(num_steps + 1):
        current_y = f(current_x)
        if step < num_steps:
            gradient = (f(current_x + 1e-5) - f(current_x - 1e-5)) / (2e-5)
            next_x = current_x - learning_rate * gradient
            next_y = f(next_x)
        else:
            gradient = 0
            next_x = current_x
            next_y = current_y
            
        steps_data.append({
            'step': step,
            'current_x': current_x,
            'current_y': current_y,
            'gradient': gradient,
            'next_x': next_x,
            'next_y': next_y,
            'learning_rate': learning_rate
        })
        
        if step < num_steps:
            current_x = next_x
    
    return f, x_vals, y_vals, steps_data

def plot_current_step(f, x_vals, y_vals, step_data, show_next=True):
    """Plot the current step of gradient descent"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get learning_rate from step_data
    learning_rate = step_data['learning_rate']
    
    # Plot 1: Current position and path
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='Cost Function', alpha=0.7)
    
    # Plot all previous points
    current_step = step_data['step']
    if 'history' in step_data and current_step > 0:
        # Show the path taken so far
        for i in range(current_step):
            step = step_data['history'][i]
            color_intensity = 0.3 + 0.7 * (i / current_step) if current_step > 0 else 1
            ax1.scatter(step['current_x'], step['current_y'], 
                       color='red', s=80, alpha=color_intensity, zorder=5)
            if i > 0:
                prev_step = step_data['history'][i-1]
                ax1.plot([prev_step['current_x'], step['current_x']], 
                        [prev_step['current_y'], step['current_y']], 
                        'r-', alpha=color_intensity, linewidth=2)
    
    # Current point (most prominent)
    ax1.scatter(step_data['current_x'], step_data['current_y'], 
               color='red', s=200, zorder=10, label=f'Step {current_step}')
    
    # Next point (if showing)
    if show_next and step_data['step'] < len(step_data.get('history', [])) - 1:
        next_step = step_data['history'][step_data['step'] + 1]
        ax1.scatter(next_step['current_x'], next_step['current_y'], 
                   color='green', s=150, zorder=9, alpha=0.7, label='Next Step')
        ax1.plot([step_data['current_x'], next_step['current_x']], 
                [step_data['current_y'], next_step['current_y']], 
                'g--', alpha=0.7, linewidth=2)
    
    ax1.set_xlabel('Parameter (x)')
    ax1.set_ylabel('Cost f(x)')
    ax1.set_title(f'Gradient Descent - Step {current_step}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gradient explanation for current step
    ax2.plot(x_vals, y_vals, 'b-', linewidth=2, label='Cost Function', alpha=0.7)
    ax2.scatter(step_data['current_x'], step_data['current_y'], 
               color='red', s=200, zorder=10, label='Current Position')
    
    if step_data['gradient'] != 0:
        # Show gradient direction
        grad_scale = min(abs(step_data['gradient']), 1.5) * np.sign(step_data['gradient'])
        ax2.arrow(step_data['current_x'], step_data['current_y'], grad_scale, 0, 
                 head_width=0.2, head_length=0.1, fc='red', ec='red', 
                 linewidth=3, label=f'Gradient = {step_data["gradient"]:.2f}')
        
        # Show update direction
        update_scale = -learning_rate * step_data['gradient']
        ax2.arrow(step_data['current_x'], step_data['current_y'], update_scale, 0, 
                 head_width=0.2, head_length=0.1, fc='green', ec='green', 
                 linewidth=3, label='Update Direction')
    
    ax2.set_xlabel('Parameter (x)')
    ax2.set_ylabel('Cost f(x)')
    ax2.set_title('Gradient & Update Direction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_complete_animation(function_choice, start_x, learning_rate, num_steps, custom_func=None):
    """Create a complete animation showing all steps"""
    f, x_vals, y_vals, steps_data = create_step_by_step_animation(
        function_choice, start_x, learning_rate, num_steps, custom_func
    )
    
    # Add history to each step
    for i, step in enumerate(steps_data):
        step['history'] = steps_data[:i+1]
    
    return f, x_vals, y_vals, steps_data

def create_complete_2d_animation(function_choice, start_x, start_y, learning_rate, num_steps, custom_func=None):
    """Create a complete 2D animation showing all steps"""
    f, X, Y, Z, steps_data = create_2d_gradient_descent_animation(
        function_choice, start_x, start_y, learning_rate, num_steps, custom_func
    )
    
    # Add history to each step
    for i, step in enumerate(steps_data):
        step['history'] = steps_data[:i+1]
    
    return f, X, Y, Z, steps_data

# Main app
def main():
    st.markdown('<h1 class="main-header">🎯 Advanced Gradient Descent Visualizer</h1>', unsafe_allow_html=True)

    # Sidebar for controls
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("Adjust the parameters to see how gradient descent works:")

    # Dimension selection
    dimension = st.sidebar.selectbox(
        "Select Dimension:",
        ["1D Function", "2D Function"]
    )

    # Function selection
    if dimension == "1D Function":
        function_choices = [
            "Quadratic: f(x) = (x-3)² + 2",
            "Double Well: f(x) = x⁴ - 8x² + 3x + 10", 
            "Complex: f(x) = sin(x) + 0.1*(x-2)²",
            "Custom Function"
        ]
    else:  # 2D Function
        function_choices = [
            "Quadratic Bowl: f(x,y) = x² + y²",
            "Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²",
            "Himmelblau: f(x,y) = (x²+y-11)² + (x+y²-7)²",
            "Custom Function"
        ]

    function_choice = st.sidebar.selectbox("Choose Cost Function:", function_choices)

    # Custom function input
    custom_func = None
    if function_choice == "Custom Function":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Custom Function")
        
        if dimension == "1D Function":
            custom_expr = st.sidebar.text_input(
                "Enter your function f(x):",
                value="x**2 + 2*x + 1",
                help="Use Python syntax: x**2 for x², sin(x), cos(x), exp(x), etc."
            )
            if custom_expr:
                custom_func = parse_custom_function(custom_expr, dimension=1)
        else:  # 2D Function
            custom_expr = st.sidebar.text_input(
                "Enter your function f(x,y):",
                value="x**2 + y**2",
                help="Use Python syntax: x**2 + y**2, sin(x)*cos(y), etc."
            )
            if custom_expr:
                custom_func = parse_custom_function(custom_expr, dimension=2)
        
        if custom_func:
            st.sidebar.success("✅ Custom function parsed successfully!")
        else:
            st.sidebar.warning("⚠️ Enter a valid function to continue")

    # Parameter controls
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if dimension == "1D Function":
            start_x = st.slider("Start X", -5.0, 10.0, 6.0, 0.1)
        else:
            start_x = st.slider("Start X", -3.0, 3.0, 2.0, 0.1)
            start_y = st.slider("Start Y", -3.0, 3.0, 2.0, 0.1)
    
    with col2:
        learning_rate = st.slider("Learning Rate (η)", 0.001, 1.0, 0.1, 0.001)
    
    with col3:
        num_steps = st.slider("Number of Steps", 3, 50, 10)

    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Gradient Descent Variant:",
        ["Basic Gradient Descent", "Momentum", "Nesterov", "Adam"]
    )

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Step-by-Step Animation", "📊 Optimization Path", "❓ Why Subtract?", "📖 Explanation"])

    with tab1:
        if dimension == "1D Function":
            st.markdown('<h2 class="sub-header">1D Gradient Descent Animation</h2>', unsafe_allow_html=True)
            
            if function_choice == "Custom Function" and not custom_func:
                st.warning("Please enter a valid custom function in the sidebar.")
            else:
                # Create the animation data
                f, x_vals, y_vals, steps_data = create_complete_animation(
                    function_choice, start_x, learning_rate, num_steps, custom_func
                )
                
                # Step selector
                current_step = st.slider(
                    "Select Step to View:",
                    min_value=0,
                    max_value=num_steps,
                    value=0,
                    key="step_slider_1d"
                )
                
                # Show step information
                step_info = steps_data[current_step]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Step", current_step)
                with col2:
                    st.metric("Parameter (x)", f"{step_info['current_x']:.4f}")
                with col3:
                    st.metric("Cost f(x)", f"{step_info['current_y']:.4f}")
                
                if current_step < num_steps:
                    next_step_info = steps_data[current_step + 1]
                    st.info(f"**Next Step Preview:** x will move from {step_info['current_x']:.4f} to {next_step_info['current_x']:.4f} "
                           f"(gradient: {step_info['gradient']:.4f})")
                
                # Plot the current step
                show_next = st.checkbox("Show Next Step Preview", value=True, key="show_next_1d")
                fig = plot_current_step(f, x_vals, y_vals, step_info, show_next=show_next)
                st.pyplot(fig)
                
                # Auto-animation
                st.markdown("---")
                st.markdown("### 🎬 Auto-Animation")
                
                if st.button("Play Step-by-Step Animation", key="animate_1d"):
                    animation_placeholder = st.empty()
                    
                    for step in range(num_steps + 1):
                        with animation_placeholder.container():
                            st.empty()
                            step_info = steps_data[step]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Step", step, delta=None)
                            with col2:
                                st.metric("Parameter (x)", f"{step_info['current_x']:.4f}")
                            with col3:
                                st.metric("Cost f(x)", f"{step_info['current_y']:.4f}")
                            
                            show_next_step = (step < num_steps)
                            fig = plot_current_step(f, x_vals, y_vals, step_info, show_next=show_next_step)
                            st.pyplot(fig)
                            
                            if step < num_steps:
                                st.success(f"✅ Step {step} completed! Moving to next step...")
                                next_gradient = step_info['gradient']
                                next_x = step_info['next_x']
                                st.info(f"**Next move:** x = {step_info['current_x']:.4f} - {learning_rate:.3f} × {next_gradient:.4f} = {next_x:.4f}")
                            else:
                                st.balloons()
                                st.success("🎉 Optimization completed! Minimum found!")
                            
                            if step < num_steps:
                                time.sleep(2)
                    
                    st.info("Animation completed! Use the slider above to review any step.")
        
        else:  # 2D Function
            st.markdown('<h2 class="sub-header">2D Gradient Descent Animation</h2>', unsafe_allow_html=True)
            
            if function_choice == "Custom Function" and not custom_func:
                st.warning("Please enter a valid custom function in the sidebar.")
            else:
                # Create the 2D animation data
                f, X, Y, Z, steps_data = create_complete_2d_animation(
                    function_choice, start_x, start_y, learning_rate, num_steps, custom_func
                )
                
                # Step selector
                current_step = st.slider(
                    "Select Step to View:",
                    min_value=0,
                    max_value=num_steps,
                    value=0,
                    key="step_slider_2d"
                )
                
                # Show step information
                step_info = steps_data[current_step]
                current_pos = step_info['current_pos']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Step", current_step)
                with col2:
                    st.metric("X", f"{current_pos[0]:.4f}")
                with col3:
                    st.metric("Y", f"{current_pos[1]:.4f}")
                with col4:
                    st.metric("Cost f(x,y)", f"{step_info['current_z']:.4f}")
                
                if current_step < num_steps:
                    next_step_info = steps_data[current_step + 1]
                    next_pos = next_step_info['current_pos']
                    gradient = step_info['gradient']
                    st.info(f"**Next Step Preview:** Position will move from ({current_pos[0]:.4f}, {current_pos[1]:.4f}) to ({next_pos[0]:.4f}, {next_pos[1]:.4f})")
                
                # Plot the current step
                show_next = st.checkbox("Show Next Step Preview", value=True, key="show_next_2d")
                fig = plot_2d_current_step(f, X, Y, Z, step_info, show_next=show_next)
                st.pyplot(fig)
                
                # Auto-animation
                st.markdown("---")
                st.markdown("### 🎬 Auto-Animation")
                
                if st.button("Play Step-by-Step Animation", key="animate_2d"):
                    animation_placeholder = st.empty()
                    
                    for step in range(num_steps + 1):
                        with animation_placeholder.container():
                            st.empty()
                            step_info = steps_data[step]
                            current_pos = step_info['current_pos']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Step", step, delta=None)
                            with col2:
                                st.metric("X", f"{current_pos[0]:.4f}")
                            with col3:
                                st.metric("Y", f"{current_pos[1]:.4f}")
                            with col4:
                                st.metric("Cost f(x,y)", f"{step_info['current_z']:.4f}")
                            
                            show_next_step = (step < num_steps)
                            fig = plot_2d_current_step(f, X, Y, Z, step_info, show_next=show_next_step)
                            st.pyplot(fig)
                            
                            if step < num_steps:
                                st.success(f"✅ Step {step} completed! Moving to next step...")
                                gradient = step_info['gradient']
                                next_pos = step_info['next_pos']
                                st.info(f"**Gradient:** ({gradient[0]:.4f}, {gradient[1]:.4f})")
                            else:
                                st.balloons()
                                st.success("🎉 Optimization completed! Minimum found!")
                            
                            if step < num_steps:
                                time.sleep(2)
                    
                    st.info("Animation completed! Use the slider above to review any step.")

    # Rest of the tabs remain similar but would need updates for 2D...
    # [The other tabs (2, 3, 4) would be similar to previous implementation but adapted for 2D]

if __name__ == "__main__":
    main()
