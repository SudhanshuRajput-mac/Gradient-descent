import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import pandas as pd

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
</style>
""", unsafe_allow_html=True)

# Animation functions (moved to main file)
def create_gradient_descent_animation(function_choice, start_x, learning_rate, num_steps, algorithm):
    """Create gradient descent animation"""
    
    # Define the function and gradient
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
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Initialize variables for animation
    current_x = start_x
    points_history = []
    
    def animate(frame):
        nonlocal current_x, points_history
        
        ax1.clear()
        ax2.clear()
        
        # Plot the function on both subplots
        for ax in [ax1, ax2]:
            ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Cost Function')
            ax.set_xlabel('Parameter (x)')
            ax.set_ylabel('Cost f(x)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_vals[0], x_vals[-1])
            ax.set_ylim(min(y_vals)-1, max(y_vals)+1)
        
        if frame == num_steps:
            ax1.set_title('Converged!')
        else:
            ax1.set_title(f'Gradient Descent - Step {frame + 1}')
        ax2.set_title('Gradient Explanation')
        
        # Append current point based on frame
        if frame == 0:
            current_y = f(current_x)
            points_history.append((current_x, current_y))
        elif frame <= num_steps:
            current_y = f(current_x)
            points_history.append((current_x, current_y))
        
        # Plot history
        if len(points_history) > 0:
            history_x, history_y = zip(*points_history)
            ax1.plot(history_x, history_y, 'ro-', markersize=6, linewidth=2, alpha=0.7)
            ax1.scatter(current_x, current_y, color='red', s=100, zorder=5)
        
        if frame < num_steps:
            # Calculate gradient and proposed update
            h_val = 1e-5
            gradient = (f(current_x + h_val) - f(current_x - h_val)) / (2 * h_val)
            new_x = current_x - learning_rate * gradient
            
            # Show info on ax1
            ax1.text(0.05, 0.95, 
                    f'x = {current_x:.2f}\n'
                    f'f(x) = {current_y:.2f}\n'
                    f'Gradient = {gradient:.2f}\n'
                    f'New x = {new_x:.2f}', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Second plot: Gradient explanation at current
            ax2.scatter(current_x, current_y, color='red', s=100, zorder=5)
            
            # Show gradient arrow
            grad_scale = min(abs(gradient), 1.5) * np.sign(gradient)
            ax2.arrow(current_x, current_y, grad_scale, 0, 
                     head_width=0.2, head_length=0.1, fc='red', ec='red', 
                     linewidth=3, label=f'Gradient = {gradient:.2f}')
            
            # Show update direction
            update_scale = -learning_rate * gradient
            ax2.arrow(current_x, current_y, update_scale, 0, 
                     head_width=0.2, head_length=0.1, fc='green', ec='green', 
                     linewidth=3, label='Update direction')
            
            ax2.legend()
            
            # Update for next iteration
            current_x = new_x
        else:
            # Final frame - show convergence
            h_val = 1e-5
            gradient = (f(current_x + h_val) - f(current_x - h_val)) / (2 * h_val)
            ax1.text(0.05, 0.95, 
                    f'x = {current_x:.2f}\n'
                    f'f(x) = {current_y:.2f}\n'
                    f'Gradient ≈ 0 (converged)', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax2.scatter(current_x, current_y, color='red', s=100, zorder=5)
            ax2.text(0.05, 0.95, 'Minimum found!\nGradient ≈ 0', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return []
    
    # Create animation with num_steps + 1 frames to show final position
    animation = FuncAnimation(fig, animate, frames=num_steps + 1, interval=1000, repeat=False, blit=False)
    
    return fig, animation

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
    """Create a static plot showing the optimization path"""
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
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Animation", "📊 Optimization Path", "❓ Why Subtract?", "📖 Explanation"])

    with tab1:
        st.markdown('<h2 class="sub-header">Live Gradient Descent Animation</h2>', unsafe_allow_html=True)
        
        # Create animation
        fig, animation = create_gradient_descent_animation(
            function_choice, start_x, learning_rate, num_steps, algorithm
        )
        
        # Display animation as GIF
        buf = io.BytesIO()
        animation.save(buf, format='gif', writer='pillow', fps=1)
        buf.seek(0)
        
        st.image(buf.getvalue(), format='GIF')
        
        # Download button for animation
        st.download_button(
            label="📥 Download Animation as GIF",
            data=buf.getvalue(),
            file_name="gradient_descent.gif",
            mime="image/gif"
        )

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
            st.metric("Convergence Rate", f"{metrics['convergence_rate']:.4f}")
        
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
            st.markdown("### Subtract")
            <div class="info-box">
            <h3> Key Insight</h3>
            <p><strong>Gradient = Direction of STEEPEST ASCENT</strong></p>
            <p>To <strong>MINIMIZE</strong> the function, we need to go in the <strong>OPPOSITE</strong> direction!</p>
            <br>
            <p><strong>
