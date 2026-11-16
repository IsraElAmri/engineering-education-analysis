"""
Educational System Analyzer - Professional Analysis Pipeline
Handles comparative analysis of education systems with AI impact assessment.
"""

import sys
import locale
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import kruskal, mannwhitneyu
from jinja2 import Template
import warnings

# Configure UTF-8 encoding
if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Safe print function for UTF-8 compatibility
def safe_print(text: str) -> None:
    """Print with fallback encoding"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))

# Replace problematic characters with safe alternatives
SYMBOLS = {
    'check': '✓' if locale.getpreferredencoding() == 'UTF-8' else '[OK]',
    'cross': '✗' if locale.getpreferredencoding() == 'UTF-8' else '[X]',
    'arrow': '→' if locale.getpreferredencoding() == 'UTF-8' else '->',
    'euro': '€' if locale.getpreferredencoding() == 'UTF-8' else 'EUR'
}

# Safe decorator for analysis functions
def safe_analysis_wrapper(func):
    """Decorator for safe analysis execution"""
    def wrapper(*args, **kwargs):
        try:
            logging.info(f"Starting: {func.__name__}")
            result = func(*args, **kwargs)
            logging.info(f"Completed: {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            return None
    return wrapper

class EducationalSystemAnalyzer:
    """
    Professional analysis pipeline for educational systems comparison.

    This class provides a comprehensive framework for analyzing and comparing
    different educational systems, including clustering, statistical testing,
    and impact assessment of AI integration.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.

        Args:
            df: DataFrame containing educational system data
        """
        self.df = df.copy()
        self.results = {}
        self.validation_report = {}
        self.X_scaled = None
        self.logger = logging.getLogger(__name__)

    @safe_analysis_wrapper
    def preprocess(self) -> 'EducationalSystemAnalyzer':
        """Clean and prepare data for analysis."""
        try:
            # Handle missing values
            self.df = self._impute_missing()

            # Detect and handle outliers
            self.df = self._handle_outliers()

            # Normalize features for clustering
            self.X_scaled = self._normalize_features()

            self.logger.info("Preprocessing completed successfully")
            return self

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    @safe_analysis_wrapper
    def cluster_analysis(self, n_clusters: int = 3) -> 'EducationalSystemAnalyzer':
        """Perform clustering analysis."""
        try:
            if self.X_scaled is None:
                raise ValueError("Must run preprocess() before clustering")

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.df['cluster'] = kmeans.fit_predict(self.X_scaled)

            # Calculate quality metrics
            self.results['clustering'] = {
                'inertia': kmeans.inertia_,
                'silhouette': silhouette_score(self.X_scaled, self.df['cluster']),
                'n_clusters': n_clusters
            }

            self.logger.info(f"Clustering completed with {n_clusters} clusters")
            return self

        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}")
            raise

    @safe_analysis_wrapper
    def statistical_comparison(self, metrics: Optional[List[str]] = None) -> 'EducationalSystemAnalyzer':
        """Compare systems statistically."""
        try:
            if metrics is None:
                metrics = ['employability_pct', 'roi_annual', 'graduation_rate_pct']

            systems = self.df['education_system'].unique()
            self.results['statistical'] = {}

            for metric in metrics:
                if metric not in self.df.columns:
                    self.logger.warning(f"Metric {metric} not found in data, skipping")
                    continue

                # Overall test
                groups = [self.df[self.df['education_system']==s][metric].dropna().values
                         for s in systems if len(self.df[self.df['education_system']==s][metric].dropna()) > 0]

                if len(groups) < 2:
                    self.logger.warning(f"Not enough data for {metric}")
                    continue

                stat, p = kruskal(*groups)

                self.results['statistical'][metric] = {
                    'overall_p': p,
                    'significant': p < 0.05
                }

                # Pairwise comparisons
                pairwise = {}
                for i, sys1 in enumerate(systems):
                    for sys2 in systems[i+1:]:
                        g1 = self.df[self.df['education_system']==sys1][metric].dropna()
                        g2 = self.df[self.df['education_system']==sys2][metric].dropna()
                        if len(g1) > 0 and len(g2) > 0:
                            _, p = mannwhitneyu(g1, g2)
                            pairwise[f'{sys1}_vs_{sys2}'] = p

                self.results['statistical'][metric]['pairwise'] = pairwise

            self.logger.info("Statistical comparison completed")
            return self

        except Exception as e:
            self.logger.error(f"Statistical comparison failed: {str(e)}")
            raise

    @safe_analysis_wrapper
    def generate_report(self, output_file: str = 'analysis_report.html') -> 'EducationalSystemAnalyzer':
        """Generate professional HTML report."""
        try:
            template = Template(self._get_report_template())

            html = template.render(
                n_countries=len(self.df),
                n_systems=self.df['education_system'].nunique(),
                n_variables=len(self.df.columns),
                clustering=self.results.get('clustering', {}),
                statistical=self.results.get('statistical', {}),
                system_comparison=self._generate_system_comparison()
            )

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)

            safe_print(f"✓ Report saved: {output_file}")
            self.logger.info(f"Report generated: {output_file}")
            return self

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise

    def _impute_missing(self) -> pd.DataFrame:
        """Impute missing values using median strategy."""
        df = self.df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isnull().any():
                # Use median imputation
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.logger.debug(f"Imputed {df[col].isnull().sum()} missing values in {col}")

        return df

    def _handle_outliers(self) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        df = self.df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['system_code', 'cluster']:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)

        return df

    def _normalize_features(self) -> np.ndarray:
        """Normalize features for clustering."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['system_code', 'cluster']]

        # Filter out columns that are all NaN or have no variance
        valid_feature_cols = []
        for col in feature_cols:
            col_data = self.df[col].dropna()
            if len(col_data) > 1 and col_data.std() > 0:
                valid_feature_cols.append(col)

        if not valid_feature_cols:
            raise ValueError("No valid numeric features found for clustering")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[valid_feature_cols])

        # Handle any remaining NaN values (shouldn't happen but safety check)
        if np.isnan(scaled_data).any():
            # Fill NaN with 0 (mean of standardized data)
            scaled_data = np.nan_to_num(scaled_data, nan=0.0)

        return scaled_data

    def _generate_system_comparison(self) -> List[Dict]:
        """Generate system comparison data for report."""
        comparison = []
        systems = self.df['education_system'].unique()
        metrics = ['employability_pct', 'graduation_rate_pct', 'roi_annual']

        for system in systems:
            system_data = self.df[self.df['education_system'] == system]
            row = {'system': system, 'countries': ', '.join(system_data['country'].unique())}

            for metric in metrics:
                if metric in system_data.columns:
                    row[f'{metric}_mean'] = round(system_data[metric].mean(), 2)
                    row[f'{metric}_std'] = round(system_data[metric].std(), 2) if not pd.isna(system_data[metric].std()) else 0

            comparison.append(row)

        return comparison

    def _get_report_template(self) -> str:
        """Get the HTML report template."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Educational Systems Analysis Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                    font-size: 2.5em;
                }
                h2 {
                    color: #34495e;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 40px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    background: white;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }
                th {
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }
                tr:nth-child(even) {
                    background-color: #f8f9fa;
                }
                .metric-card {
                    background: #ecf0f1;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 4px solid #3498db;
                }
                .significant {
                    background-color: #d4edda;
                    color: #155724;
                    padding: 2px 6px;
                    border-radius: 3px;
                }
                .not-significant {
                    background-color: #f8d7da;
                    color: #721c24;
                    padding: 2px 6px;
                    border-radius: 3px;
                }
                .footer {
                    text-align: center;
                    margin-top: 40px;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Educational Systems Comparative Analysis</h1>

                <h2>📊 Data Summary</h2>
                <div class="metric-card">
                    <strong>Total Countries:</strong> {{ n_countries }}<br>
                    <strong>Educational Systems:</strong> {{ n_systems }}<br>
                    <strong>Variables Analyzed:</strong> {{ n_variables }}
                </div>

                <h2>🎯 Clustering Results</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
                    <tr>
                        <td>Silhouette Score</td>
                        <td>{{ clustering.silhouette|round(3) }}</td>
                        <td class="{{ 'significant' if clustering.silhouette > 0.5 else 'not-significant' }}">
                            {{ 'Good separation' if clustering.silhouette > 0.5 else 'Poor separation' }}
                        </td>
                    </tr>
                    <tr>
                        <td>Inertia</td>
                        <td>{{ clustering.inertia|round(2) }}</td>
                        <td>Lower values indicate tighter clusters</td>
                    </tr>
                </table>

                <h2>📈 Statistical Comparisons</h2>
                {% for metric, results in statistical.items() %}
                <h3>{{ metric.replace('_', ' ').title() }}</h3>
                <table>
                    <tr>
                        <th>Test</th>
                        <th>P-value</th>
                        <th>Significant?</th>
                    </tr>
                    <tr>
                        <td>Overall (Kruskal-Wallis)</td>
                        <td>{{ results.overall_p|round(4) }}</td>
                        <td class="{{ 'significant' if results.significant else 'not-significant' }}">
                            {{ 'Yes' if results.significant else 'No' }}
                        </td>
                    </tr>
                </table>
                {% endfor %}

                <h2>🏆 System Comparison</h2>
                <table>
                    <tr>
                        <th>System</th>
                        <th>Countries</th>
                        <th>Employability (%)</th>
                        <th>Graduation Rate (%)</th>
                        <th>ROI</th>
                    </tr>
                    {% for system in system_comparison %}
                    <tr>
                        <td>{{ system.system }}</td>
                        <td>{{ system.countries }}</td>
                        <td>{{ system.employability_pct_mean }} ± {{ system.employability_pct_std }}</td>
                        <td>{{ system.graduation_rate_pct_mean }} ± {{ system.graduation_rate_pct_std }}</td>
                        <td>{{ system.roi_annual_mean }} ± {{ system.roi_annual_std }}</td>
                    </tr>
                    {% endfor %}
                </table>

                <h2>💡 Recommendations</h2>
                <ul>
                    <li>Sample size should be increased for more robust conclusions</li>
                    <li>Longitudinal data would enable causal inference</li>
                    <li>Additional socioeconomic controls recommended</li>
                    <li>Consider AI integration impact on system performance</li>
                </ul>

                <div class="footer">
                    <p>Report generated by Educational System Analyzer</p>
                </div>
            </div>
        </body>
        </html>
        """

class DataLoader:
    """Utility class for loading and preprocessing educational data."""

    @staticmethod
    def load_datasets(before_path: str, after_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load before and after AI datasets."""
        try:
            df_before = pd.read_csv(before_path, sep=';', encoding='utf-8')
            df_after = pd.read_csv(after_path, encoding='utf-8')
            logging.info(f"Loaded datasets: before({df_before.shape}), after({df_after.shape})")
            return df_before, df_after
        except Exception as e:
            logging.error(f"Failed to load datasets: {str(e)}")
            raise

    @staticmethod
    def standardize_countries(df: pd.DataFrame, country_mapping: Dict[str, str]) -> pd.DataFrame:
        """Standardize country names."""
        df = df.copy()
        df['country'] = df['country'].str.strip()
        df['country'] = df['country'].replace(country_mapping)
        return df

    @staticmethod
    def clean_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Clean and preprocess a dataset."""
        df_clean = df.copy()

        # Convert numeric columns
        numeric_cols = []
        for col in df_clean.columns:
            if col not in ['country', 'degree_type', 'main_sectors', 'estimate_notes']:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    numeric_cols.append(col)
                except:
                    pass

        # Clean percentages
        pct_keywords = ['pct', 'rate', 'ratio', 'satisfaction', 'employability', 'graduation']
        for col in df_clean.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in pct_keywords):
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].clip(lower=0, upper=100)

        # Clean selective admission
        if 'selective_admission' in df_clean.columns:
            df_clean['selective_admission_clean'] = df_clean['selective_admission'].astype(str)
            df_clean['is_selective'] = df_clean['selective_admission_clean'].str.contains(
                'Yes|yes|True|true|Oui|oui|Moderate|competitive',
                case=False, na=False
            ).astype(int)

        safe_print(f"  Dataset '{dataset_name}' cleaned: {df_clean.shape}")
        return df_clean

    @staticmethod
    def prepare_phase1_data(df_clean: pd.DataFrame, system_classification: Dict[str, List[str]]) -> pd.DataFrame:
        """Prepare data for system comparison analysis."""
        df_phase = df_clean.copy()

        # Add system classification
        df_phase['education_system'] = None
        df_phase['system_code'] = None

        system_codes = {
            'Système Français (Grandes Écoles)': 1,
            'Système Allemand/Europe Centrale': 2,
            'Système Anglo-Saxon': 3
        }

        for system, countries in system_classification.items():
            mask = df_phase['country'].isin(countries)
            df_phase.loc[mask, 'education_system'] = system
            df_phase.loc[mask, 'system_code'] = system_codes[system]

        # Select relevant columns
        relevant_cols = [
            'country', 'education_system', 'system_code',
            'duration_years', 'degree_type',
            'graduation_rate_pct', 'employability_pct',
            'avg_start_salary_eur', 'annual_cost_local_eur',
            'student_satisfaction_pct', 'tech_infra_score_0_100',
            'pct_public_funding', 'gdp_per_capita_usd',
            'human_development_index', 'student_prof_ratio',
            'is_selective'
        ]

        available_cols = [col for col in relevant_cols if col in df_phase.columns]
        df_phase_filtered = df_phase[available_cols].copy()

        # Calculate derived metrics
        if all(col in df_phase_filtered.columns for col in ['avg_start_salary_eur', 'annual_cost_local_eur']):
            df_phase_filtered['roi_annual'] = (
                df_phase_filtered['avg_start_salary_eur'] /
                df_phase_filtered['annual_cost_local_eur'].replace(0, np.nan)
            ).round(2)

        if all(col in df_phase_filtered.columns for col in [
            'graduation_rate_pct', 'employability_pct', 'student_satisfaction_pct'
        ]):
            df_phase_filtered['performance_score'] = df_phase_filtered[[
                'graduation_rate_pct', 'employability_pct', 'student_satisfaction_pct'
            ]].mean(axis=1).round(2)

        return df_phase_filtered

class AIAnalysis:
    """Class for analyzing AI impact on educational systems."""

    def __init__(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        self.df_before = df_before.copy()
        self.df_after = df_after.copy()
        self.df_combined = None

    @safe_analysis_wrapper
    def prepare_ai_analysis(self, system_classification: Dict[str, List[str]]) -> pd.DataFrame:
        """Prepare data for AI impact analysis."""
        # Find common columns
        common_cols = list(set(self.df_before.columns) & set(self.df_after.columns))
        common_cols = [col for col in common_cols if col not in ['country', 'degree_type', 'main_sectors', 'estimate_notes']]

        # Select key metrics
        key_metrics = [
            'graduation_rate_pct', 'employability_pct', 'avg_start_salary_eur',
            'market_demand_index_0_100', 'pct_courses_AI', 'tech_infra_score_0_100', 'rd_pct_gdp'
        ]
        selected_metrics = [m for m in key_metrics if m in common_cols]

        # Create before/after datasets
        df_before_sel = self.df_before[['country'] + selected_metrics].copy()
        df_after_sel = self.df_after[['country'] + selected_metrics].copy()

        # Rename columns
        df_before_sel = df_before_sel.add_suffix('_before').rename(columns={'country_before': 'country'})
        df_after_sel = df_after_sel.add_suffix('_after').rename(columns={'country_after': 'country'})

        # Merge
        df_combined = pd.merge(df_before_sel, df_after_sel, on='country', how='inner')

        # Calculate changes
        for metric in selected_metrics:
            before_col = f'{metric}_before'
            after_col = f'{metric}_after'

            if before_col in df_combined.columns and after_col in df_combined.columns:
                # Absolute change
                df_combined[f'{metric}_change'] = df_combined[after_col] - df_combined[before_col]

                # Percentage change
                df_combined[f'{metric}_change_pct'] = (
                    (df_combined[after_col] - df_combined[before_col]) /
                    df_combined[before_col].replace(0, np.nan)
                ) * 100

        # Add system classification
        df_combined['education_system'] = None
        for system, countries in system_classification.items():
            mask = df_combined['country'].isin(countries)
            df_combined.loc[mask, 'education_system'] = system

        # Categorize AI adoption
        if 'pct_courses_AI_after' in df_combined.columns:
            df_combined['AI_adoption_level'] = pd.cut(
                df_combined['pct_courses_AI_after'],
                bins=[-1, 0, 15, 30, 100],
                labels=['None', 'Low', 'Medium', 'High']
            )

        # Calculate improvement score
        change_cols = [col for col in df_combined.columns if '_change' in col and '_pct' not in col]
        if change_cols:
            df_combined['improvement_score'] = df_combined[change_cols].mean(axis=1)

        self.df_combined = df_combined
        return df_combined

    def get_ai_impact_summary(self) -> Dict:
        """Get summary of AI impact analysis."""
        if self.df_combined is None:
            raise ValueError("Must run prepare_ai_analysis first")

        summary = {}

        # AI introduction
        if 'pct_courses_AI_change' in self.df_combined.columns:
            ai_intro = self.df_combined[['country', 'education_system', 'pct_courses_AI_before', 'pct_courses_AI_after', 'pct_courses_AI_change']].copy()
            ai_intro['AI_introduced'] = ai_intro['pct_courses_AI_change'] > 0
            summary['ai_introduction'] = ai_intro

        # Impact by system
        if all(col in self.df_combined.columns for col in ['education_system', 'employability_pct_change', 'pct_courses_AI_change']):
            impact_by_system = self.df_combined.groupby('education_system').agg({
                'employability_pct_change': ['mean', 'std', 'count'],
                'pct_courses_AI_change': 'mean'
            }).round(2)
            summary['impact_by_system'] = impact_by_system

        # Correlation
        corr_cols = ['pct_courses_AI_change', 'employability_pct_change', 'graduation_rate_pct_change']
        available_corr_cols = [col for col in corr_cols if col in self.df_combined.columns]
        if available_corr_cols:
            correlation = self.df_combined[available_corr_cols].corr().round(3)
            summary['correlation'] = correlation

        return summary


class Visualization:
    """Class for generating educational system visualizations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logging.getLogger(__name__)

    @safe_analysis_wrapper
    def generate_comprehensive_visualizations(self, output_dir: str = '../Downloads/sof1/') -> None:
        """Generate comprehensive visualizations for educational systems analysis."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier

        # Prepare data for ML and visualization
        df_ml = self.df.copy()

        # Handle missing values
        numeric_cols = df_ml.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_ml[col].isnull().any():
                df_ml[col] = df_ml.groupby('education_system')[col].transform(lambda x: x.fillna(x.median()))
                if df_ml[col].isnull().any():
                    df_ml[col].fillna(df_ml[col].median(), inplace=True)

        # Encode systems
        label_encoder = LabelEncoder()
        df_ml['system_encoded'] = label_encoder.fit_transform(df_ml['education_system'])

        # Select features
        features_ml = [
            'graduation_rate_pct', 'employability_pct', 'avg_start_salary_eur',
            'annual_cost_local_eur', 'student_satisfaction_pct', 'tech_infra_score_0_100',
            'pct_public_funding', 'gdp_per_capita_usd', 'human_development_index',
            'student_prof_ratio', 'roi_annual', 'performance_score'
        ]
        features_available = [col for col in df_ml.columns if col in features_ml]

        # Normalize features
        scaler = StandardScaler()
        X = df_ml[features_available]
        X_scaled = scaler.fit_transform(X)
        y = df_ml['system_encoded']

        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        df_ml['cluster'] = clusters

        # PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(16, 6))

        # Plot 1: By Education System
        plt.subplot(1, 3, 1)
        colors = ['red', 'blue', 'green']
        for i, system in enumerate(label_encoder.classes_):
            mask = df_ml['education_system'] == system
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=colors[i], label=system, s=200, alpha=0.7)

            # Annotate countries
            for idx, row in df_ml[mask].iterrows():
                plt.annotate(row['country'],
                            (X_pca[idx, 0], X_pca[idx, 1]),
                            fontsize=9, fontweight='bold',
                            xytext=(5, 5), textcoords='offset points')

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('PCA - Real Education Systems')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: By Clusters
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=clusters, cmap='viridis', s=200, alpha=0.7)

        for idx, row in df_ml.iterrows():
            plt.annotate(row['country'],
                        (X_pca[idx, 0], X_pca[idx, 1]),
                        fontsize=9, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('PCA - K-Means Clusters')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)

        # Plot 3: Feature Importance
        plt.subplot(1, 3, 3)
        model_full = RandomForestClassifier(random_state=42, n_estimators=100)
        model_full.fit(X_scaled, y)

        importance_df = pd.DataFrame({
            'variable': features_available,
            'importance': model_full.feature_importances_
        }).sort_values('importance', ascending=False)

        top_n = min(8, len(importance_df))
        plt.barh(range(top_n), importance_df['importance'].head(top_n)[::-1])
        plt.yticks(range(top_n), importance_df['variable'].head(top_n)[::-1])
        plt.xlabel('Importance')
        plt.title('Top Feature Importances')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir + 'comprehensive_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()  # Don't show plot in script mode

        # Generate classification results
        results, importance_df = self._generate_classification_results(df_ml, X_scaled, y, label_encoder)

        # Classification Results Plot
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        countries = [r['country'] for r in results]
        correctness = [1 if r['correct'] else 0 for r in results]
        colors_plot = ['green' if c else 'red' for c in correctness]

        bars = plt.bar(countries, correctness, color=colors_plot)
        plt.ylabel('Correct (1) / Incorrect (0)')
        plt.title('Classification Results by Country')
        plt.ylim([0, 1.2])

        for bar, result in zip(bars, results):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                     '✓' if result['correct'] else '✗',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.xticks(rotation=45, ha='right')

        # System Comparison Plot
        plt.subplot(1, 2, 2)
        systems = df_ml['education_system'].unique()
        system_means = []
        for system in systems:
            system_data = df_ml[df_ml['education_system'] == system]
            system_means.append({
                'system': system,
                'employability': system_data['employability_pct'].mean(),
                'graduation': system_data['graduation_rate_pct'].mean(),
                'salary': system_data['avg_start_salary_eur'].mean() / 1000,  # Convert to thousands
                'cost': system_data['annual_cost_local_eur'].mean() / 1000
            })

        x = np.arange(len(systems))
        width = 0.2

        plt.bar(x - width*1.5, [s['employability'] for s in system_means], width, label='Employability %', alpha=0.8)
        plt.bar(x - width/2, [s['graduation'] for s in system_means], width, label='Graduation %', alpha=0.8)
        plt.bar(x + width/2, [s['salary'] for s in system_means], width, label='Salary (k€)', alpha=0.8)
        plt.bar(x + width*1.5, [s['cost'] for s in system_means], width, label='Cost (k€)', alpha=0.8)

        plt.xlabel('Education Systems')
        plt.ylabel('Values')
        plt.title('System Comparison Metrics')
        plt.xticks(x, [s['system'][:20] + '...' if len(s['system']) > 20 else s['system'] for s in system_means], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir + 'classification_and_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        safe_print("✓ Visualizations generated and saved!")
        return results, importance_df

    def _generate_classification_results(self, df_ml: pd.DataFrame, X_scaled: np.ndarray,
                                       y: np.ndarray, label_encoder: LabelEncoder) -> Tuple[List[Dict], pd.DataFrame]:
        """Generate classification results for leave-one-out cross-validation."""
        from sklearn.ensemble import RandomForestClassifier

        results = []
        for i, test_country in enumerate(df_ml['country'].tolist()):
            train_indices = df_ml[df_ml['country'] != test_country].index.tolist()
            test_indices = df_ml[df_ml['country'] == test_country].index.tolist()

            if not test_indices:
                continue

            X_train = X_scaled[train_indices]
            X_test = X_scaled[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            true_system = df_ml.loc[test_indices[0], 'education_system']
            pred_encoded = y_pred[0]
            system_mapping = {
                0: 'Système Français (Grandes Écoles)',
                1: 'Système Allemand/Europe Centrale',
                2: 'Système Anglo-Saxon'
            }
            pred_system = system_mapping.get(pred_encoded, "Unknown")
            correct = (true_system == pred_system)

            results.append({
                'country': test_country,
                'true_system': true_system,
                'pred_system': pred_system,
                'correct': correct
            })

        # Get feature importance from last model
        importance_df = pd.DataFrame({
            'variable': [col for col in df_ml.columns if col in [
                'graduation_rate_pct', 'employability_pct', 'avg_start_salary_eur',
                'annual_cost_local_eur', 'student_satisfaction_pct', 'tech_infra_score_0_100',
                'pct_public_funding', 'gdp_per_capita_usd', 'human_development_index',
                'student_prof_ratio', 'roi_annual', 'performance_score'
            ]],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return results, importance_df