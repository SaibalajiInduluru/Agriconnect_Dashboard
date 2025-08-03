import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Agri Connect Dashboard", 
    page_icon="🌱", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Agri Connect branding
st.markdown("""
<style>
    .main {
        padding: 2rem 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #2e7d32, #4caf50);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .section-header {
        background: linear-gradient(90deg, #66bb6a, #81c784);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e8f5e8;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e8f5e8;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, #f1f8e9, #e8f5e8);
        border-right: 3px solid #4caf50;
    }
    
    .sidebar-header {
        background: linear-gradient(90deg, #2e7d32, #4caf50);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #c8e6c9;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-metric {
        text-align: center;
        padding: 0.8rem;
        background: #f8fff8;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #2e7d32;
        display: block;
        margin-bottom: 0.2rem;
    }
    
    .sidebar-metric-label {
        font-size: 0.9rem;
        color: #555;
        font-weight: 500;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #c8e6c9;
    }
    
    .stFileUploader > div {
        background-color: white;
        border: 2px dashed #4caf50;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .success-message {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data(uploaded_file):
    """Load data from uploaded file with error handling"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'txt':
            df = pd.read_csv(uploaded_file, sep='\t')
        else:
            st.error(f"❌ Unsupported file format: {file_extension}")
            return None
            
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def detect_date_columns(df):
    """Detect potential date/time columns"""
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                sample_data = df[col].dropna().head(10)
                if len(sample_data) > 0:
                    pd.to_datetime(sample_data, errors='raise', infer_datetime_format=True)
                    date_columns.append(col)
            except:
                continue
    return date_columns

def get_numeric_columns(df):
    """Get numeric columns from DataFrame"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    """Get categorical columns from DataFrame"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def create_pie_chart(df, column):
    """Create a smaller, professional pie chart"""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    value_counts = df[column].value_counts().head(8)
    
    colors = ['#2e7d32', '#4caf50', '#66bb6a', '#81c784', '#a5d6a7', '#c8e6c9', '#e1f5fe', '#f3e5f5']
    
    wedges, texts, autotexts = ax.pie(
        value_counts.values, 
        labels=value_counts.index, 
        autopct='%1.1f%%',
        colors=colors[:len(value_counts)],
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(9)
    
    ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold', pad=15, color='#2e7d32')
    plt.tight_layout()
    
    return fig

def create_bar_chart(df, numeric_col, categorical_col):
    """Create professional bar chart"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    grouped_data = df.groupby(categorical_col)[numeric_col].mean().sort_values(ascending=False)
    
    if len(grouped_data) > 12:
        grouped_data = grouped_data.head(12)
        st.info("📊 Showing top 12 categories for better visualization")
    
    bars = ax.bar(range(len(grouped_data)), grouped_data.values, 
                  color='#4caf50', alpha=0.8, edgecolor='#2e7d32', linewidth=1)
    
    ax.set_xlabel(categorical_col, fontsize=12, fontweight='bold', color='#2e7d32')
    ax.set_ylabel(f'Average {numeric_col}', fontsize=12, fontweight='bold', color='#2e7d32')
    ax.set_title(f'Average {numeric_col} by {categorical_col}', 
                fontsize=14, fontweight='bold', pad=15, color='#2e7d32')
    ax.set_xticks(range(len(grouped_data)))
    ax.set_xticklabels(grouped_data.index, rotation=45, ha='right', fontsize=10)
    
    for bar, value in zip(bars, grouped_data.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    plt.tight_layout()
    
    return fig

def create_correlation_heatmap(df):
    """Create professional correlation heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdYlGn', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                ax=ax,
                linewidths=0.5)
    
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=15, color='#2e7d32')
    plt.tight_layout()
    
    return fig

def create_sidebar_mini_chart(df, column, chart_type='bar'):
    """Create mini charts for sidebar"""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    
    if chart_type == 'bar' and column in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[column].value_counts().head(5)
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='#4caf50', alpha=0.7)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'Top 5 {column}', fontsize=10, color='#2e7d32', fontweight='bold')
        
    elif chart_type == 'hist' and column in df.select_dtypes(include=[np.number]).columns:
        ax.hist(df[column].dropna(), bins=10, color='#4caf50', alpha=0.7, edgecolor='#2e7d32')
        ax.set_title(f'{column} Distribution', fontsize=10, color='#2e7d32', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid(alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    return fig

def create_multi_column_comparison(df, columns, chart_type='bar'):
    """Create comparison charts for multiple columns"""
    if chart_type == 'bar' and len(columns) >= 2:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar chart for categorical columns
        if all(col in get_categorical_columns(df) for col in columns):
            # Get common categories across columns
            all_categories = set()
            for col in columns[:3]:  # Limit to 3 columns
                all_categories.update(df[col].value_counts().head(6).index)
            common_categories = list(all_categories)[:8]
            
            x = np.arange(len(common_categories))
            width = 0.8 / len(columns[:3])  # Adjust width based on number of columns
            colors = ['#4caf50', '#66bb6a', '#81c784', '#a5d6a7']
            
            for i, col in enumerate(columns[:3]):
                values = [df[df[col] == cat].shape[0] for cat in common_categories]
                bars = ax.bar(x + i * width, values, width, 
                             label=col, alpha=0.8, color=colors[i])
            
            ax.set_xlabel('Categories', fontweight='bold', color='#2e7d32')
            ax.set_ylabel('Count', fontweight='bold', color='#2e7d32')
            ax.set_title(f'Multi-Column Comparison: {", ".join(columns[:3])}', 
                        fontweight='bold', color='#2e7d32', fontsize=14)
            ax.set_xticks(x + width)
            ax.set_xticklabels(common_categories, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
        # Create side-by-side comparison for numeric columns
        elif all(col in get_numeric_columns(df) for col in columns):
            x = np.arange(len(columns))
            values = [df[col].mean() for col in columns]
            colors = ['#4caf50', '#66bb6a', '#81c784', '#a5d6a7'][:len(columns)]
            
            bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='#2e7d32', linewidth=1)
            
            ax.set_xlabel('Columns', fontweight='bold', color='#2e7d32')
            ax.set_ylabel('Average Values', fontweight='bold', color='#2e7d32')
            ax.set_title(f'Average Comparison: {", ".join(columns)}', 
                        fontweight='bold', color='#2e7d32', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(columns, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    return None

def create_multi_category_visualization(df, columns):
    """Create multiple visualizations for categorical columns"""
    if len(columns) < 2:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Category Analysis Dashboard', fontsize=16, fontweight='bold', color='#2e7d32')
    
    colors_palette = ['#4caf50', '#66bb6a', '#81c784', '#a5d6a7', '#c8e6c9']
    
    # Top-left: Pie chart for first column
    if len(columns) >= 1:
        value_counts1 = df[columns[0]].value_counts().head(6)
        wedges, texts, autotexts = axes[0,0].pie(value_counts1.values, 
                                                labels=value_counts1.index, 
                                                autopct='%1.1f%%',
                                                colors=colors_palette,
                                                startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(8)
        axes[0,0].set_title(f'Distribution: {columns[0]}', fontweight='bold', color='#2e7d32')
    
    # Top-right: Bar chart for second column
    if len(columns) >= 2:
        value_counts2 = df[columns[1]].value_counts().head(8)
        bars = axes[0,1].bar(range(len(value_counts2)), value_counts2.values, 
                            color='#66bb6a', alpha=0.8, edgecolor='#2e7d32')
        axes[0,1].set_xlabel(columns[1], fontweight='bold', color='#2e7d32')
        axes[0,1].set_ylabel('Count', fontweight='bold', color='#2e7d32')
        axes[0,1].set_title(f'Frequency: {columns[1]}', fontweight='bold', color='#2e7d32')
        axes[0,1].set_xticks(range(len(value_counts2)))
        axes[0,1].set_xticklabels(value_counts2.index, rotation=45, ha='right')
        axes[0,1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, value_counts2.values):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Bottom-left: Horizontal bar for third column (if available)
    if len(columns) >= 3:
        value_counts3 = df[columns[2]].value_counts().head(6)
        bars = axes[1,0].barh(range(len(value_counts3)), value_counts3.values, 
                             color='#81c784', alpha=0.8, edgecolor='#2e7d32')
        axes[1,0].set_ylabel(columns[2], fontweight='bold', color='#2e7d32')
        axes[1,0].set_xlabel('Count', fontweight='bold', color='#2e7d32')
        axes[1,0].set_title(f'Horizontal View: {columns[2]}', fontweight='bold', color='#2e7d32')
        axes[1,0].set_yticks(range(len(value_counts3)))
        axes[1,0].set_yticklabels(value_counts3.index)
        axes[1,0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, value_counts3.values):
            width = bar.get_width()
            axes[1,0].text(width, bar.get_y() + bar.get_height()/2.,
                          f'{value}', ha='left', va='center', fontweight='bold', fontsize=9)
    else:
        axes[1,0].axis('off')
        axes[1,0].text(0.5, 0.5, 'Select 3rd column\nfor additional view', 
                      ha='center', va='center', transform=axes[1,0].transAxes,
                      fontsize=12, color='#666')
    
    # Bottom-right: Comparison of top categories across columns
    if len(columns) >= 2:
        # Get top 5 categories from each column
        top_cats = {}
        for col in columns[:3]:
            top_cats[col] = df[col].value_counts().head(5)
        
        # Create a comparison chart
        x_pos = np.arange(5)
        width = 0.25
        
        for i, (col, counts) in enumerate(top_cats.items()):
            if i < 3:  # Limit to 3 columns
                values = list(counts.values)[:5]
                # Pad with zeros if less than 5 categories
                while len(values) < 5:
                    values.append(0)
                
                axes[1,1].bar(x_pos + i*width, values, width, 
                             label=col, alpha=0.8, color=colors_palette[i])
        
        axes[1,1].set_xlabel('Top Categories (Rank)', fontweight='bold', color='#2e7d32')
        axes[1,1].set_ylabel('Count', fontweight='bold', color='#2e7d32')
        axes[1,1].set_title('Top 5 Categories Comparison', fontweight='bold', color='#2e7d32')
        axes[1,1].set_xticks(x_pos + width)
        axes[1,1].set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
        axes[1,1].legend()
        axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig
    """Create time series chart for date columns"""
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.sort_values(date_col)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df_copy[date_col], df_copy[numeric_col], 
               color='#4caf50', linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        ax.fill_between(df_copy[date_col], df_copy[numeric_col], 
                       alpha=0.3, color='#c8e6c9')
        
        ax.set_xlabel(date_col, fontsize=12, fontweight='bold', color='#2e7d32')
        ax.set_ylabel(numeric_col, fontsize=12, fontweight='bold', color='#2e7d32')
        ax.set_title(f'{numeric_col} Over Time', fontsize=14, fontweight='bold', pad=15, color='#2e7d32')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    except:
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🌱 Agri Connect</div>
        <div class="main-subtitle">Professional Data Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🌱 Agri Connect Control Panel</div>', unsafe_allow_html=True)
        
        # File Upload Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**📁 Data Upload**")
        uploaded_file = st.file_uploader(
            "Choose your data file", 
            type=['csv', 'xlsx', 'xls', 'txt'],
            help="Supported: CSV, Excel, Tab-delimited TXT"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            # Quick Data Overview
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**📊 Quick Overview**")
            
            # Load data for sidebar metrics
            df_preview = load_data(uploaded_file)
            if df_preview is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'''
                    <div class="sidebar-metric">
                        <div class="sidebar-metric-value">{len(df_preview):,}</div>
                        <div class="sidebar-metric-label">Rows</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="sidebar-metric">
                        <div class="sidebar-metric-value">{len(df_preview.columns)}</div>
                        <div class="sidebar-metric-label">Columns</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Quick visualization in sidebar
                numeric_cols = get_numeric_columns(df_preview)
                categorical_cols = get_categorical_columns(df_preview)
                
                if categorical_cols:
                    st.markdown("**📈 Quick Chart**")
                    selected_sidebar_col = st.selectbox(
                        "Preview column:", 
                        categorical_cols[:5], 
                        key="sidebar_preview",
                        help="Quick preview of data distribution"
                    )
                    
                    if selected_sidebar_col:
                        fig = create_sidebar_mini_chart(df_preview, selected_sidebar_col, 'bar')
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis Options
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**⚙️ Display Options**")
            show_raw_data = st.checkbox("📋 Show raw data", value=False)
            show_summary = st.checkbox("📊 Show statistics", value=True)
            show_insights = st.checkbox("💡 Show insights", value=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization Controls
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**📈 Chart Controls**")
            chart_size = st.select_slider(
                "Chart size:",
                options=["Small", "Medium", "Large"],
                value="Medium"
            )
            show_values = st.checkbox("Show values on charts", value=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Help Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**❓ Help & Tips**")
        with st.expander("📖 How to use"):
            st.markdown("""
            **Getting Started:**
            1. Upload your data file above
            2. Configure display options
            3. Explore the dashboard
            
            **Supported Files:**
            - CSV files (.csv)
            - Excel files (.xlsx, .xls)
            - Text files (.txt)
            
            **Features:**
            - Multi-column comparisons
            - Interactive visualizations
            - Statistical analysis
            - Trend analysis
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("🔄 Loading and processing data..."):
            df = load_data(uploaded_file)
        
        if df is not None:
            st.markdown("""
            <div class="success-message">
                ✅ Data loaded successfully! Ready for analysis.
            </div>
            """, unsafe_allow_html=True)
            
            # Performance Overview
            st.markdown('<div class="section-header">📊 Performance Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{len(df):,}</div>
                    <div class="metric-label">Total Records</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{len(df.columns)}</div>
                    <div class="metric-label">Data Fields</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{missing_pct:.1f}%</div>
                    <div class="metric-label">Missing Data</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{memory_mb:.1f} MB</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Raw Data
            if show_raw_data:
                st.markdown('<div class="section-header">🔍 Raw Data Explorer</div>', unsafe_allow_html=True)
                with st.expander("📋 View Complete Dataset", expanded=False):
                    st.dataframe(df, use_container_width=True, height=400)
            
            # Statistical Summary
            if show_summary:
                st.markdown('<div class="section-header">📈 Statistical Summary</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**📋 Data Types & Quality**")
                    info_df = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null': df.count(),
                        'Null Count': df.isnull().sum(),
                        'Null %': (df.isnull().sum() / len(df) * 100).round(1)
                    })
                    st.dataframe(info_df, use_container_width=True)
                
                with col2:
                    st.markdown("**📊 Descriptive Statistics**")
                    st.dataframe(df.describe(include='all').round(2), use_container_width=True)
            
            # Get column types
            numeric_cols = get_numeric_columns(df)
            categorical_cols = get_categorical_columns(df)
            date_cols = detect_date_columns(df)
            
            # Multi-Column Analysis
            st.markdown('<div class="section-header">🔀 Multi-Column Analysis</div>', unsafe_allow_html=True)
            
            # Separate tabs for different analysis types
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📊 Numeric Comparison", "🎯 Scatter Analysis", "🏷️ Category Dashboard"])
            
            with analysis_tab1:
                st.markdown("**📊 Multi-Column Numeric Comparison**")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    available_cols = numeric_cols + categorical_cols
                    selected_comparison_cols = st.multiselect(
                        "Select columns to compare:",
                        available_cols,
                        max_selections=4,
                        help="Choose columns for side-by-side comparison"
                    )
                
                with col2:
                    if len(selected_comparison_cols) >= 2:
                        fig = create_multi_column_comparison(df, selected_comparison_cols, "bar")
                        if fig:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("Select at least 2 columns to see comparison")
            
            with analysis_tab2:
                st.markdown("**🎯 Advanced Scatter Plot Analysis**")
                
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        selected_scatter_cols = st.multiselect(
                            "Select numeric columns for scatter analysis:",
                            numeric_cols,
                            default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
                            max_selections=4,
                            help="Choose 2-4 numeric columns for scatter plot analysis"
                        )
                        
                        if len(selected_scatter_cols) >= 2:
                            show_correlation = st.checkbox("Show correlation values", value=True)
                            show_trendline = st.checkbox("Show trend lines", value=True)
                            
                            # Display data info
                            st.markdown("**📊 Data Summary:**")
                            for col in selected_scatter_cols[:2]:
                                valid_count = df[col].count()
                                total_count = len(df)
                                st.write(f"• **{col}**: {valid_count}/{total_count} valid values")
                    
                    with col2:
                        if len(selected_scatter_cols) >= 2:
                            # Create scatter plot
                            try:
                                fig = create_multi_column_comparison(df, selected_scatter_cols, "scatter")
                                if fig:
                                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close()
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.error("Unable to create scatter plot. Please check your data.")
                            except Exception as e:
                                st.error(f"Error creating scatter plot: {str(e)}")
                                st.info("This might be due to non-numeric data or insufficient valid data points.")
                            
                            # Show correlation matrix for selected columns
                            if show_correlation and len(selected_scatter_cols) >= 2:
                                st.markdown("**🔢 Correlation Matrix**")
                                try:
                                    # Clean data for correlation calculation
                                    clean_data = df[selected_scatter_cols].replace([np.inf, -np.inf], np.nan).dropna()
                                    if len(clean_data) > 1:
                                        corr_matrix = clean_data.corr()
                                        st.dataframe(corr_matrix.round(3), use_container_width=True)
                                    else:
                                        st.warning("Insufficient data for correlation calculation.")
                                except Exception as e:
                                    st.error(f"Error calculating correlations: {str(e)}")
                        else:
                            st.info("Select at least 2 numeric columns for scatter analysis")
                else:
                    st.warning("⚠️ No numeric columns found in your data. Scatter plots require numeric data.")
                    if len(get_categorical_columns(df)) > 0:
                        st.info("💡 Try the Category Dashboard tab for analyzing your categorical data.")

            
            with analysis_tab3:
                st.markdown("**🏷️ Multi-Category Visualization Dashboard**")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    selected_category_cols = st.multiselect(
                        "Select categorical columns:",
                        categorical_cols,
                        max_selections=3,
                        help="Choose 2-3 categorical columns for comprehensive analysis"
                    )
                    
                    if len(selected_category_cols) >= 2:
                        st.markdown("**📋 Quick Stats:**")
                        for col in selected_category_cols:
                            unique_count = df[col].nunique()
                            most_common = df[col].mode().iloc[0] if not df[col].empty else "N/A"
                            st.write(f"• **{col}**: {unique_count} categories, most common: '{most_common}'")
                
                with col2:
                    if len(selected_category_cols) >= 2:
                        fig = create_multi_category_visualization(df, selected_category_cols)
                        if fig:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("Select at least 2 categorical columns for dashboard view")
            
            # Enhanced Visualizations Section
            st.markdown('<div class="section-header">📊 Enhanced Data Visualizations</div>', unsafe_allow_html=True)
            
            # Chart layout
            col1, col2 = st.columns([1, 2])
            
            # Pie Chart
            if categorical_cols:
                with col1:
                    st.markdown("**🥧 Category Distribution**")
                    selected_cat_col = st.selectbox(
                        "Select categorical column:",
                        categorical_cols,
                        key="pie_chart"
                    )
                    
                    if selected_cat_col:
                        unique_count = df[selected_cat_col].nunique()
                        if unique_count > 15:
                            st.warning(f"⚠️ {unique_count} unique values detected. Showing top 8.")
                        
                        with st.container():
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            fig = create_pie_chart(df, selected_cat_col)
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                            st.markdown('</div>', unsafe_allow_html=True)
            
            # Bar Chart
            if numeric_cols and categorical_cols:
                with col2:
                    st.markdown("**📊 Comparative Analysis**")
                    subcol1, subcol2 = st.columns(2)
                    
                    with subcol1:
                        selected_numeric = st.selectbox(
                            "Numeric column:",
                            numeric_cols,
                            key="bar_numeric"
                        )
                    
                    with subcol2:
                        selected_categorical = st.selectbox(
                            "Group by:",
                            categorical_cols,
                            key="bar_categorical"
                        )
                    
                    if selected_numeric and selected_categorical:
                        with st.container():
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            fig = create_bar_chart(df, selected_numeric, selected_categorical)
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                            st.markdown('</div>', unsafe_allow_html=True)
            
            # Full width charts
            # Correlation Heatmap
            if len(numeric_cols) >= 2:
                st.markdown("**🔥 Correlation Analysis**")
                with st.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = create_correlation_heatmap(df)
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Time Series Analysis
            if date_cols and numeric_cols:
                st.markdown('<div class="section-header">📈 Trends Analysis</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_date = st.selectbox("Select date column:", date_cols, key="time_date")
                with col2:
                    selected_numeric_time = st.selectbox("Select numeric column:", numeric_cols, key="time_numeric")
                
                if selected_date and selected_numeric_time:
                    with st.container():
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        fig = create_time_series_chart(df, selected_date, selected_numeric_time)
                        if fig:
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                        else:
                            st.warning("⚠️ Unable to create time series chart with selected columns")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced Analytics Section
            if len(numeric_cols) >= 2:
                st.markdown('<div class="section-header">🔬 Advanced Analytics</div>', unsafe_allow_html=True)
                
                tab1, tab2, tab3 = st.tabs(["📈 Correlation Matrix", "📊 Distribution Analysis", "🎯 Outlier Detection"])
                
                with tab1:
                    st.markdown("**🔥 Detailed Correlation Analysis**")
                    correlation_cols = st.multiselect(
                        "Select columns for correlation:",
                        numeric_cols,
                        default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
                        help="Choose numeric columns to analyze correlations"
                    )
                    
                    if len(correlation_cols) >= 2:
                        fig = create_correlation_heatmap(df[correlation_cols])
                        if fig:
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                            
                            # Show strongest correlations
                            corr_matrix = df[correlation_cols].corr()
                            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                            corr_matrix = corr_matrix.mask(mask).abs()
                            
                            strongest_corr = corr_matrix.unstack().sort_values(ascending=False).head(3)
                            st.markdown("**🔍 Strongest Correlations:**")
                            for (col1, col2), value in strongest_corr.items():
                                if not pd.isna(value):
                                    st.write(f"• {col1} ↔ {col2}: {value:.3f}")
                
                with tab2:
                    st.markdown("**📊 Distribution Comparison**")
                    dist_col1, dist_col2 = st.columns(2)
                    
                    with dist_col1:
                        dist_column1 = st.selectbox("First column:", numeric_cols, key="dist1")
                    with dist_col2:
                        dist_column2 = st.selectbox("Second column:", numeric_cols, key="dist2")
                    
                    if dist_column1 and dist_column2 and dist_column1 != dist_column2:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # First distribution
                        ax1.hist(df[dist_column1].dropna(), bins=20, color='#4caf50', alpha=0.7, edgecolor='#2e7d32')
                        ax1.set_title(f'{dist_column1} Distribution', fontweight='bold', color='#2e7d32')
                        ax1.grid(alpha=0.3)
                        
                        # Second distribution
                        ax2.hist(df[dist_column2].dropna(), bins=20, color='#66bb6a', alpha=0.7, edgecolor='#2e7d32')
                        ax2.set_title(f'{dist_column2} Distribution', fontweight='bold', color='#2e7d32')
                        ax2.grid(alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                
                with tab3:
                    st.markdown("**🎯 Outlier Analysis**")
                    outlier_column = st.selectbox("Select column for outlier detection:", numeric_cols, key="outlier")
                    
                    if outlier_column:
                        Q1 = df[outlier_column].quantile(0.25)
                        Q3 = df[outlier_column].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = df[(df[outlier_column] < (Q1 - 1.5 * IQR)) | 
                                     (df[outlier_column] > (Q3 + 1.5 * IQR))]
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.boxplot(df[outlier_column].dropna(), patch_artist=True, 
                                      boxprops=dict(facecolor='#c8e6c9', alpha=0.7))
                            ax.set_title(f'Box Plot: {outlier_column}', fontweight='bold', color='#2e7d32')
                            ax.grid(alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                        
                        with col2:
                            st.markdown("**📊 Outlier Summary**")
                            st.metric("Total Outliers", len(outliers))
                            st.metric("Outlier %", f"{len(outliers)/len(df)*100:.1f}%")
                            
                            if len(outliers) > 0:
                                st.markdown("**Top Outliers:**")
                                extreme_outliers = outliers.nlargest(3, outlier_column)[outlier_column]
                                for idx, value in extreme_outliers.items():
                                    st.write(f"Row {idx}: {value:.2f}")

            # Insights section (enhanced)
            if show_insights:
                st.markdown('<div class="section-header">💡 Data Insights & Recommendations</div>', unsafe_allow_html=True)
            
                insights_col1, insights_col2, insights_col3, insights_col4 = st.columns(4)
                
                with insights_col1:
                    if numeric_cols:
                        highest_variance_col = df[numeric_cols].var().idxmax()
                        variance_value = df[numeric_cols].var().max()
                        st.info(f"📊 **Highest Variance**\n{highest_variance_col}\n({variance_value:.2f})")
                
                with insights_col2:
                    if categorical_cols:
                        most_categories = max([(col, df[col].nunique()) for col in categorical_cols], key=lambda x: x[1])
                        st.info(f"🏷️ **Most Diverse**\n{most_categories[0]}\n({most_categories[1]} categories)")
                
                with insights_col3:
                    completeness = ((df.count() / len(df)) * 100).mean()
                    color = "🟢" if completeness > 90 else "🟡" if completeness > 70 else "🔴"
                    st.info(f"{color} **Completeness**\nOverall Data\n({completeness:.1f}%)")
                
                with insights_col4:
                    if numeric_cols and len(numeric_cols) >= 2:
                        corr_matrix = df[numeric_cols].corr().abs()
                        max_corr = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).max().max()
                        st.info(f"🔗 **Max Correlation**\nBetween Variables\n({max_corr:.3f})")
                
                # Detailed recommendations
                st.markdown("### 🎯 Recommendations")
                recommendations = []
                
                if completeness < 80:
                    recommendations.append("🔍 **Data Quality**: Consider investigating missing data patterns")
                
                if len(numeric_cols) >= 3:
                    recommendations.append("📊 **Analysis Opportunity**: Multiple numeric columns available for advanced statistical analysis")
                
                if date_cols:
                    recommendations.append("📈 **Trend Analysis**: Time-based data detected - consider temporal analysis")
                
                if len(categorical_cols) > 0 and any(df[col].nunique() > 20 for col in categorical_cols):
                    recommendations.append("🏷️ **Categorization**: Some categorical columns have many unique values - consider grouping")
                
                for rec in recommendations:
                    st.markdown(f"• {rec}")
                
                if not recommendations:
                    st.success("✅ Your data appears to be well-structured and ready for analysis!")
    
    else:
        st.markdown('<div class="section-header">👋 Welcome to Agri Connect Dashboard</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🚀 Getting Started
        
        Upload your data file using the sidebar to begin your analysis journey. Our dashboard supports:
        
        - **📄 CSV Files** - Comma-separated values
        - **📊 Excel Files** - Both .xlsx and .xls formats  
        - **📋 Text Files** - Tab-delimited data
        
        ### 📈 What You'll Get
        
        - **Performance Overview** - Key metrics at a glance
        - **Statistical Analysis** - Comprehensive data summary
        - **Interactive Visualizations** - Charts and graphs
        - **Trend Analysis** - Time-based insights
        - **Data Quality Assessment** - Missing data analysis
        
        Ready to unlock insights from your agricultural data? Upload a file to get started! 🌱
        """)

if __name__ == "__main__":
    main()