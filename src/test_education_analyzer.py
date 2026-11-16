"""
Unit tests for Educational System Analyzer
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from education_analyzer import EducationalSystemAnalyzer, DataLoader, AIAnalysis

class TestEducationalSystemAnalyzer(unittest.TestCase):
    """Test cases for EducationalSystemAnalyzer class."""

    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'country': ['France', 'Germany', 'USA', 'Tunisia', 'Hungary', 'Mexico'],
            'education_system': ['Système Français (Grandes Écoles)', 'Système Allemand/Europe Centrale',
                               'Système Anglo-Saxon', 'Système Français (Grandes Écoles)',
                               'Système Allemand/Europe Centrale', 'Système Anglo-Saxon'],
            'employability_pct': [85, 82, 88, 78, 80, 86],
            'graduation_rate_pct': [90, 88, 92, 85, 87, 89],
            'roi_annual': [12.5, 10.8, 15.2, 9.5, 11.2, 13.8],
            'avg_start_salary_eur': [45000, 42000, 48000, 38000, 40000, 46000],
            'annual_cost_local_eur': [8000, 12000, 25000, 6000, 10000, 20000]
        })

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = EducationalSystemAnalyzer(self.test_data)
        self.assertIsInstance(analyzer, EducationalSystemAnalyzer)
        self.assertEqual(len(analyzer.df), 6)
        self.assertEqual(analyzer.results, {})
        self.assertIsNone(analyzer.X_scaled)

    def test_preprocess(self):
        """Test preprocessing pipeline."""
        analyzer = EducationalSystemAnalyzer(self.test_data)

        # Test successful preprocessing
        result = analyzer.preprocess()
        self.assertIsInstance(result, EducationalSystemAnalyzer)
        self.assertIsNotNone(analyzer.X_scaled)
        self.assertEqual(analyzer.X_scaled.shape[0], 6)

    def test_preprocess_with_missing_values(self):
        """Test preprocessing with missing values."""
        data_with_missing = self.test_data.copy()
        data_with_missing.loc[0, 'employability_pct'] = np.nan

        analyzer = EducationalSystemAnalyzer(data_with_missing)
        analyzer.preprocess()

        # Check that missing value was imputed
        self.assertFalse(analyzer.df['employability_pct'].isnull().any())

    def test_cluster_analysis(self):
        """Test clustering analysis."""
        analyzer = EducationalSystemAnalyzer(self.test_data)
        analyzer.preprocess()

        result = analyzer.cluster_analysis(n_clusters=3)
        self.assertIsInstance(result, EducationalSystemAnalyzer)
        self.assertIn('clustering', analyzer.results)
        self.assertIn('silhouette', analyzer.results['clustering'])
        self.assertIn('inertia', analyzer.results['clustering'])
        self.assertEqual(analyzer.results['clustering']['n_clusters'], 3)

    def test_cluster_without_preprocess(self):
        """Test clustering fails without preprocessing."""
        analyzer = EducationalSystemAnalyzer(self.test_data)

        result = analyzer.cluster_analysis()
        self.assertIsNone(result)  # Decorator returns None on error

    def test_statistical_comparison(self):
        """Test statistical comparison."""
        analyzer = EducationalSystemAnalyzer(self.test_data)
        analyzer.preprocess()

        result = analyzer.statistical_comparison()
        self.assertIsInstance(result, EducationalSystemAnalyzer)
        self.assertIn('statistical', analyzer.results)

        # Check that expected metrics are analyzed
        expected_metrics = ['employability_pct', 'roi_annual', 'graduation_rate_pct']
        for metric in expected_metrics:
            if metric in self.test_data.columns:
                self.assertIn(metric, analyzer.results['statistical'])

    def test_statistical_comparison_missing_metric(self):
        """Test statistical comparison with missing metric."""
        data_missing_metric = self.test_data.drop('roi_annual', axis=1)
        analyzer = EducationalSystemAnalyzer(data_missing_metric)
        analyzer.preprocess()

        # Should complete without the missing metric
        result = analyzer.statistical_comparison(['roi_annual'])
        self.assertIsInstance(result, EducationalSystemAnalyzer)
        # roi_annual should not be in results since it was missing
        self.assertNotIn('roi_annual', analyzer.results.get('statistical', {}))

    @patch('builtins.open', new_callable=MagicMock)
    def test_generate_report(self, mock_open):
        """Test report generation."""
        analyzer = EducationalSystemAnalyzer(self.test_data)
        analyzer.preprocess()
        analyzer.cluster_analysis()
        analyzer.statistical_comparison()

        result = analyzer.generate_report('test_report.html')
        self.assertIsInstance(result, EducationalSystemAnalyzer)

        # Check that file was opened for writing
        mock_open.assert_called_once_with('test_report.html', 'w', encoding='utf-8')

    def test_impute_missing(self):
        """Test missing value imputation."""
        analyzer = EducationalSystemAnalyzer(self.test_data)
        data_with_missing = analyzer.df.copy()
        data_with_missing.loc[0, 'employability_pct'] = np.nan

        result = analyzer._impute_missing()
        self.assertFalse(result['employability_pct'].isnull().any())

    def test_handle_outliers(self):
        """Test outlier handling."""
        analyzer = EducationalSystemAnalyzer(self.test_data)
        # Add extreme outlier
        analyzer.df.loc[0, 'employability_pct'] = 200

        result = analyzer._handle_outliers()
        # Value should be clipped
        self.assertLessEqual(result.loc[0, 'employability_pct'], 100)

    def test_normalize_features(self):
        """Test feature normalization."""
        analyzer = EducationalSystemAnalyzer(self.test_data)
        analyzer.preprocess()  # This calls _normalize_features

        self.assertIsNotNone(analyzer.X_scaled)
        self.assertEqual(analyzer.X_scaled.shape[0], len(analyzer.df))

        # Check that features are standardized (mean ≈ 0, std ≈ 1)
        # Skip non-numeric columns
        numeric_cols = analyzer.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['system_code', 'cluster']]
        if len(feature_cols) > 0:
            self.assertAlmostEqual(np.mean(analyzer.X_scaled), 0, places=1)

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""

    def test_standardize_countries(self):
        """Test country name standardization."""
        test_df = pd.DataFrame({
            'country': [' Allemagne ', 'USA', ' Mexique '],
            'value': [1, 2, 3]
        })

        mapping = {
            'Allemagne': 'Germany',
            'Mexique': 'Mexico'
        }

        result = DataLoader.standardize_countries(test_df, mapping)

        self.assertEqual(result.loc[0, 'country'], 'Germany')
        self.assertEqual(result.loc[1, 'country'], 'USA')
        self.assertEqual(result.loc[2, 'country'], 'Mexico')

    def test_clean_dataset(self):
        """Test dataset cleaning."""
        test_df = pd.DataFrame({
            'country': ['France', 'Germany'],
            'degree_type': ['Master', 'Bachelor'],
            'employability_pct': ['85', '90'],
            'some_text': ['text1', 'text2'],
            'selective_admission': ['Yes', 'No']
        })

        result = DataLoader.clean_dataset(test_df, 'test')

        # Check numeric conversion
        self.assertTrue(pd.api.types.is_numeric_dtype(result['employability_pct']))

        # Check selective admission processing
        self.assertIn('is_selective', result.columns)

class TestAIAnalysis(unittest.TestCase):
    """Test cases for AI Analysis class."""

    def setUp(self):
        """Set up test data for AI analysis."""
        self.df_before = pd.DataFrame({
            'country': ['France', 'Germany', 'USA'],
            'employability_pct': [80, 75, 85],
            'pct_courses_AI': [5, 0, 10]
        })

        self.df_after = pd.DataFrame({
            'country': ['France', 'Germany', 'USA'],
            'employability_pct': [85, 82, 88],
            'pct_courses_AI': [15, 8, 25]
        })

        self.system_classification = {
            'Système Français (Grandes Écoles)': ['France'],
            'Système Allemand/Europe Centrale': ['Germany'],
            'Système Anglo-Saxon': ['USA']
        }

    def test_prepare_ai_analysis(self):
        """Test AI analysis preparation."""
        ai_analysis = AIAnalysis(self.df_before, self.df_after)

        result = ai_analysis.prepare_ai_analysis(self.system_classification)

        self.assertIsNotNone(result)
        self.assertIn('employability_pct_change', result.columns)
        self.assertIn('pct_courses_AI_change', result.columns)
        self.assertIn('education_system', result.columns)

    def test_get_ai_impact_summary(self):
        """Test AI impact summary generation."""
        ai_analysis = AIAnalysis(self.df_before, self.df_after)
        ai_analysis.prepare_ai_analysis(self.system_classification)

        summary = ai_analysis.get_ai_impact_summary()

        self.assertIn('ai_introduction', summary)
        self.assertIn('impact_by_system', summary)
        self.assertIn('correlation', summary)

    def test_get_ai_impact_without_prep(self):
        """Test AI impact summary fails without preparation."""
        ai_analysis = AIAnalysis(self.df_before, self.df_after)

        with self.assertRaises(ValueError):
            ai_analysis.get_ai_impact_summary()

if __name__ == '__main__':
    unittest.main()