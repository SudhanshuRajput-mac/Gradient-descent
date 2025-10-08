import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import pandas as pd
import base64
import time
learning_rate = 0.02
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
</style>
""", unsafe_allow_html=True)

def create_step_by_step_animation(function_choice, start_x, learning_rate, num_steps):
    """Create step-by-step animation showing progression from start to finish"""
    
    # Define the function
    if function_choice == "Quadratic: f(x) = (x-3)² + 2":
        def f(x):
            return (x - 3)**2 + 2
        x_vals = np.linspace(-1, 7, 400)
        
    elif function_choice == "Double Well: f(x) = x⁴ - 8x² + 3x + 10":
        def f(x):
            return x**4 - 8*x**2 + 3*x + 10
        x_vals = np.linspace(-3, 4, 400)
        
    else:  # Complex function
        def f(x):
            return np.sin(x) + 0.1*(x-2)**2
        x_vals = np.linspace(-1, 5, 400)
    
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
            'next_y': next_y
        })
        
        if step < num_steps:
            current_x = next_x
    
    return f, x_vals, y_vals, steps_data

def plot_current_step(f, x_vals, y_vals, step_data, show_next=True):
    """Plot the current step of gradient descent"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Current position and path
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='Cost Function', alpha=0.7)
    
    # Plot all previous points
    current_step = step_data['step']
    if current_step > 0:
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
    if show_next and step_data['step'] < len(step_data['history']) - 1:
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

def create_complete_animation(function_choice, start_x, learning_rate, num_steps):
    """Create a complete animation showing all steps"""
    f, x_vals, y_vals, steps_data = create_step_by_step_animation(
        function_choice, start_x, learning_rate, num_steps
    )
    
    # Add history to each step
    for i, step in enumerate(steps_data):
        step['history'] = steps_data[:i+1]
    
    return f, x_vals, y_vals, steps_data

def plot_why_subtract(start_x, learning_rate):
    """Create the 'why subtract gradient' visualization"""
    
    def f(x):
        return (x - 3)**2 + 2
    
    def df(x):
        return 2 * (x - 3)
    
    x_vals = np.linspace(0, 6, 100)
    y_vals = f(x_vals)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the function
    ax.plot(x_vals, y_vals, 'b-', linewidth=3, alpha=0.7, label='Cost Function f(x)')
    
    # Starting point
    start_y = f(start_x)
    gradient = df(start_x)
    
    # Plot starting point
    ax.scatter(start_x, start_y, color='red', s=200, zorder=5, label='Current Position')
    
    # Show gradient direction (points uphill)
    ax.arrow(start_x, start_y, 0.8, 0, head_width=0.2, head_length=0.2, 
             fc='red', ec='red', linewidth=4, label=f'Gradient = {gradient:.2f} (Points UPHILL)')
    
    # Show what happens if we SUBTRACT gradient (correct - downhill)
    correct_x = start_x - learning_rate * gradient
    correct_y = f(correct_x)
    ax.arrow(start_x, start_y, -learning_rate * gradient, 0, head_width=0.2, head_length=0.2,
             fc='green', ec='green', linewidth=4, label='SUBTRACT Gradient (DOWNHILL - CORRECT)')
    ax.scatter(correct_x, correct_y, color='green', s=200, zorder=5)
    
    # Show what happens if we ADD gradient (wrong - uphill)
    wrong_x = start_x + learning_rate * gradient
    wrong_y = f(wrong_x)
    ax.arrow(start_x, start_y, learning_rate * gradient, 0, head_width=0.2, head_length=0.2,
             fc='orange', ec='orange', linewidth=4, linestyle='--', label='ADD Gradient (UPHILL - WRONG)')
    ax.scatter(wrong_x, wrong_y, color='orange', s=200, zorder=5)
    
    # Add annotations
    ax.annotate('Higher Cost!', (wrong_x, wrong_y), xytext=(wrong_x+0.3, wrong_y+1),
                arrowprops=dict(arrowstyle='->', color='orange'), fontsize=12, color='orange')
    
    ax.annotate('Lower Cost!', (correct_x, correct_y), xytext=(correct_x-1.5, correct_y+1),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=12, color='green')
    
    # Mathematical explanation
    equation_text = (
        "MATHEMATICAL REASON:\n\n"
        "Gradient ∇f(x) points in direction\nof STEEPEST ASCENT (uphill)\n\n"
        "To MINIMIZE the function, we go in\nthe OPPOSITE direction:\n\n"
        "UPDATE RULE:\n"
        f"x_new = x_old - η × ∇f(x)\n"
        f"       = {start_x:.1f} - {learning_rate} × {gradient:.1f}\n"
        f"       = {correct_x:.1f}"
    )
    
    ax.text(0.02, 0.98, equation_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Parameter Value (x)')
    ax.set_ylabel('Cost f(x)')
    ax.set_title('WHY We Subtract The Gradient in Gradient Descent')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return fig

# Utility functions
def get_function(function_choice):
    """Get the function based on selection"""
    if function_choice == "Quadratic: f(x) = (x-3)² + 2":
        return lambda x: (x - 3)**2 + 2
    elif function_choice == "Double Well: f(x) = x⁴ - 8x² + 3x + 10":
        return lambda x: x**4 - 8*x**2 + 3*x + 10
    else:  # Complex function
        return lambda x: np.sin(x) + 0.1*(x-2)**2

def calculate_gradient(function_choice, x, h=1e-5):
    """Calculate gradient numerically"""
    f = get_function(function_choice)
    return (f(x + h) - f(x - h)) / (2 * h)

def create_static_optimization_plot(function_choice, start_x, learning_rate, num_steps, algorithm):
    """Create a static plot showing the entire optimization path"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    f = get_function(function_choice)
    
    if function_choice == "Quadratic: f(x) = (x-3)² + 2":
        x_vals = np.linspace(-1, 7, 400)
    elif function_choice == "Double Well: f(x) = x⁴ - 8x² + 3x + 10":
        x_vals = np.linspace(-3, 4, 400)
    else:  # Complex function
        x_vals = np.linspace(-1, 5, 400)
    
    y_vals = f(x_vals)
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Cost Function')
    
    # Simulate gradient descent
    current_x = start_x
    x_history = [current_x]
    y_history = [f(current_x)]
    
    for step in range(num_steps):
        gradient = calculate_gradient(function_choice, current_x)
        current_x = current_x - learning_rate * gradient
        x_history.append(current_x)
        y_history.append(f(current_x))
    
    ax.plot(x_history, y_history, 'ro-', linewidth=2, markersize=6, label='Optimization Path')
    ax.set_xlabel('Parameter (x)')
    ax.set_ylabel('Cost f(x)')
    ax.set_title(f'Gradient Descent Path - {algorithm}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def calculate_convergence_metrics(function_choice, start_x, learning_rate, num_steps):
    """Calculate convergence metrics"""
    f = get_function(function_choice)
    current_x = start_x
    initial_cost = f(current_x)
    
    for step in range(num_steps):
        gradient = calculate_gradient(function_choice, current_x)
        current_x = current_x - learning_rate * gradient
    
    final_cost = f(current_x)
    
    return {
        'final_cost': final_cost,
        'improvement': initial_cost - final_cost,
        'convergence_rate': (initial_cost - final_cost) / initial_cost if initial_cost != 0 else 0
    }

def get_parameter_history(function_choice, start_x, learning_rate, num_steps):
    """Get parameter values at each step"""
    f = get_function(function_choice)
    current_x = start_x
    history = []
    
    for step in range(num_steps + 1):
        cost = f(current_x)
        gradient = calculate_gradient(function_choice, current_x) if step < num_steps else 0
        history.append({
            'Step': step,
            'Parameter (x)': round(current_x, 4),
            'Cost f(x)': round(cost, 4),
            'Gradient': round(gradient, 4)
        })
        
        if step < num_steps:
            current_x = current_x - learning_rate * gradient
    
    return pd.DataFrame(history)

# Main app
def main():
    st.markdown('<h1 class="main-header">🎯 Gradient Descent Visualizer</h1>', unsafe_allow_html=True)

    # Sidebar for controls
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("Adjust the parameters to see how gradient descent works:")

    # Function selection
    function_choice = st.sidebar.selectbox(
        "Choose Cost Function:",
        ["Quadratic: f(x) = (x-3)² + 2", 
         "Double Well: f(x) = x⁴ - 8x² + 3x + 10",
         "Complex: f(x) = sin(x) + 0.1*(x-2)²"]
    )

    # Parameter controls
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        start_x = st.slider("Start X", -2.0, 8.0, 6.0, 0.1)
    with col2:
        learning_rate = st.slider("Learning Rate (η)", 0.01, 0.8, 0.3, 0.01)
    with col3:
        num_steps = st.slider("Number of Steps", 3, 20, 8)

    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Gradient Descent Variant:",
        ["Basic Gradient Descent", "Momentum", "Nesterov", "Adam"]
    )

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Step-by-Step Animation", "📊 Optimization Path", "❓ Why Subtract?", "📖 Explanation"])

    with tab1:
        st.markdown('<h2 class="sub-header">Step-by-Step Gradient Descent Animation</h2>', unsafe_allow_html=True)
        
        # Create the animation data
        f, x_vals, y_vals, steps_data = create_complete_animation(
            function_choice, start_x, learning_rate, num_steps
        )
        
        # Step selector
        current_step = st.slider(
            "Select Step to View:",
            min_value=0,
            max_value=num_steps,
            value=0,
            key="step_slider"
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
        show_next = st.checkbox("Show Next Step Preview", value=True)
        fig = plot_current_step(f, x_vals, y_vals, step_info, show_next=show_next)
        st.pyplot(fig)
        
        # Auto-animation
        st.markdown("---")
        st.markdown("### 🎬 Auto-Animation")
        
        if st.button("Play Step-by-Step Animation"):
            placeholder = st.empty()
            for step in range(num_steps + 1):
                with placeholder.container():
                    step_info = steps_data[step]
                    
                    # Update metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Step", step)
                    with col2:
                        st.metric("Parameter (x)", f"{step_info['current_x']:.4f}")
                    with col3:
                        st.metric("Cost f(x)", f"{step_info['current_y']:.4f}")
                    
                    # Plot
                    fig = plot_current_step(f, x_vals, y_vals, step_info, show_next=(step < num_steps))
                    st.pyplot(fig)
                    
                    if step < num_steps:
                        st.success(f"✅ Step {step} completed! Moving to next step...")
                    else:
                        st.balloons()
                        st.success("🎉 Optimization completed! Minimum found!")
                    
                    # Add a small delay between steps
                    time.sleep(2)
            
            st.info("Animation completed! Use the slider above to review any step.")

    with tab2:
        st.markdown('<h2 class="sub-header">Optimization Path Analysis</h2>', unsafe_allow_html=True)
        
        # Create static plot showing the entire optimization path
        fig_static = create_static_optimization_plot(function_choice, start_x, learning_rate, num_steps, algorithm)
        st.pyplot(fig_static)
        
        # Convergence analysis
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Convergence Metrics")
            # Calculate and display convergence metrics
            metrics = calculate_convergence_metrics(function_choice, start_x, learning_rate, num_steps)
            st.metric("Final Cost", f"{metrics['final_cost']:.4f}")
            st.metric("Total Improvement", f"{metrics['improvement']:.4f}")
            st.metric("Convergence Rate", f"{metrics['convergence_rate']:.2%}")
        
        with col2:
            st.markdown("### 🎯 Parameter History")
            # Show parameter values at each step
            history = get_parameter_history(function_choice, start_x, learning_rate, num_steps)
            st.dataframe(history)

    with tab3:
        st.markdown('<h2 class="sub-header">Why Do We Subtract The Gradient?</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create the "why subtract" visualization
            fig_why = plot_why_subtract(start_x, learning_rate)
            st.pyplot(fig_why)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h3>🎯 Key Insight</h3>
            <p><strong>Gradient = Direction of STEEPEST ASCENT</strong></p>
            <p>To <strong>MINIMIZE</strong> the function, we need to go in the <strong>OPPOSITE</strong> direction!</p>
            <br>
            <p><strong>Update Rule:</strong></p>
            <p><code>x_new = x_old - η × ∇f(x)</code></p>
            <br>
            <p>• <span style="color: green">SUBTRACT</span> → Move DOWNHILL ✅</p>
            <p>• <span style="color: red">ADD</span> → Move UPHILL ❌</p>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### 📖 Gradient Descent Explained")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### What is Gradient Descent?
            
            Gradient Descent is an **optimization algorithm** used to find the minimum of a function. 
            In machine learning, we use it to minimize the **cost function** and find the best parameters for our model.
            
            #### 🎯 The Core Idea
            
            1. **Start** at a random point on the cost function
            2. **Calculate** the gradient (slope) at that point
            3. **Move** in the opposite direction of the gradient
            4. **Repeat** until you reach the minimum
            
            #### ⚙️ Key Components
            
            - **Cost Function**: Measures how wrong our predictions are
            - **Gradient**: Direction of steepest ascent
            - **Learning Rate**: Size of each step
            - **Parameters**: Values we're optimizing
            """)
        
        with col2:
            st.markdown("""
            #### 📝 Mathematical Foundation
            
            **Update Rule:**
            ```
            θ = θ - η * ∇J(θ)
            ```
            
            Where:
            - `θ` = Parameters
            - `η` = Learning rate
            - `∇J(θ)` = Gradient of cost function
            
            #### 🚀 Why Subtract?
            
            The gradient points **uphill** (direction of steepest ascent).
            
            To **minimize** the function, we need to go **downhill**, so we:
            - **SUBTRACT** the gradient to go downhill ✅
            - Never add the gradient (that would go uphill) ❌
            
            #### 💡 Pro Tips
            
            - Normalize your features
            - Monitor learning curves
            - Use learning rate scheduling
            - Try different optimizers
            """)
        
        # Interactive formula explanation
        st.markdown("---")
        st.markdown("### 🧮 Interactive Formula Explorer")
        
        current_x = st.slider("Current x value", -2.0, 8.0, 5.0, 0.1, key="formula_x")
        current_lr = st.slider("Current learning rate", 0.01, 1.0, 0.3, 0.01, key="formula_lr")
        
        # Calculate and display the formula step by step
        f = get_function(function_choice)
        gradient = calculate_gradient(function_choice, current_x)
        new_x = current_x - current_lr * gradient
        
        st.latex(f"\\text{{Gradient }}\\nabla f(x) = {gradient:.3f}")
        st.latex(f"x_{{new}} = x_{{old}} - \\eta \\times \\nabla f(x)")
        st.latex(f"x_{{new}} = {current_x:.3f} - {current_lr:.3f} \\times {gradient:.3f} = {new_x:.3f}")
        
        st.metric("Cost Improvement", f"{f(current_x) - f(new_x):.4f}")

if __name__ == "__main__":
    main()
