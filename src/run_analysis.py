#!/usr/bin/env python3
"""
Educational Systems Analysis - Complete Pipeline
Run this script to perform the complete analysis with all visualizations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from education_analyzer import (
    EducationalSystemAnalyzer,
    DataLoader,
    AIAnalysis,
    safe_print
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main analysis pipeline."""
    safe_print("=" * 60)
    safe_print("🎓 EDUCATIONAL SYSTEMS COMPARATIVE ANALYSIS")
    safe_print("=" * 60)

    # ==================== DATA LOADING ====================
    safe_print("\n📂 LOADING DATASETS...")

    try:
        import os
        before_path = 'data/sample_before_ai.csv' if os.path.exists('data/sample_before_ai.csv') else ('../data/sample_before_ai.csv' if os.path.exists('../data/sample_before_ai.csv') else 'sample_before_ai.csv')
        after_path = 'data/sample_after_ai.csv' if os.path.exists('data/sample_after_ai.csv') else ('../data/sample_after_ai.csv' if os.path.exists('../data/sample_after_ai.csv') else 'sample_after_ai.csv')
        df_before, df_after = DataLoader.load_datasets(before_path, after_path)
        safe_print(f"✓ Loaded datasets: before({df_before.shape}), after({df_after.shape})")
    except Exception as e:
        safe_print(f"✗ Failed to load datasets: {e}")
        return

    # ==================== DATA CLEANING ====================
    safe_print("\n🧹 CLEANING DATASETS...")

    country_mapping = {
        'Allemagne': 'Germany',
        'USA': 'United States',
        'Mexique': 'Mexico',
        'Hongrie': 'Hungary',
        'Tunisie': 'Tunisia'
    }

    df_before = DataLoader.standardize_countries(df_before, country_mapping)
    df_after = DataLoader.standardize_countries(df_after, country_mapping)

    df_before_clean = DataLoader.clean_dataset(df_before, "before AI")
    df_after_clean = DataLoader.clean_dataset(df_after, "after AI")

    # ==================== SYSTEM CLASSIFICATION ====================
    safe_print("\n🏫 DEFINING EDUCATIONAL SYSTEMS...")

    system_classification = {
        'Système Français (Grandes Écoles)': ['France', 'Tunisia'],
        'Système Allemand/Europe Centrale': ['Germany', 'Hungary'],
        'Système Anglo-Saxon': ['United States', 'Mexico']
    }

    safe_print("Educational system classification:")
    for system, countries in system_classification.items():
        safe_print(f"  {system}: {countries}")

    # ==================== PHASE 1: SYSTEM COMPARISON ====================
    safe_print("\n📊 PREPARING PHASE 1 DATA...")

    def prepare_phase1_data(df_clean):
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

    df_phase1 = prepare_phase1_data(df_after_clean)
    safe_print(f"✓ Phase 1 dataset prepared: {df_phase1.shape}")

    # ==================== RUN ANALYSIS ====================
    safe_print("\n🔬 RUNNING COMPREHENSIVE ANALYSIS...")

    analyzer = EducationalSystemAnalyzer(df_phase1)
    analyzer.preprocess().cluster_analysis().statistical_comparison()

    # ==================== PHASE 2: AI IMPACT ====================
    safe_print("\n🤖 PREPARING PHASE 2 DATA (AI IMPACT)...")

    ai_analysis = AIAnalysis(df_before_clean, df_after_clean)
    df_phase2 = ai_analysis.prepare_ai_analysis(system_classification)
    safe_print(f"✓ Phase 2 dataset prepared: {df_phase2.shape}")

    # ==================== VISUALIZATIONS ====================
    safe_print("\n📈 GENERATING VISUALIZATIONS...")

    # Prepare data for visualization
    df_ml = df_phase1.copy()

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
    output_dir = '../Downloads/sof1/'
    plt.savefig(output_dir + 'comprehensive_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()  # Don't show plot in script mode

    # Classification Results
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

    # ==================== GENERATE REPORT ====================
    safe_print("\n📋 GENERATING PROFESSIONAL REPORT...")

    analyzer.generate_report(output_dir + 'phase1_analysis_report.html')
    safe_print("✓ Professional HTML report generated!")

    # ==================== SAVE DATA ====================
    safe_print("\n💾 SAVING PROCESSED DATA...")

    df_phase1.to_csv(output_dir + 'phase1_comparaison_systemes_educatifs.csv', index=False, encoding='utf-8')
    df_phase2.to_csv(output_dir + 'phase2_impact_IA_avant_apres.csv', index=False, encoding='utf-8')

    # Create system comparison summary
    system_comparison_data = []
    comparison_metrics = [
        'graduation_rate_pct', 'employability_pct', 'avg_start_salary_eur',
        'annual_cost_local_eur', 'student_satisfaction_pct', 'roi_annual',
        'performance_score', 'tech_infra_score_0_100', 'pct_public_funding'
    ]

    for system in df_phase1['education_system'].unique():
        system_data = df_phase1[df_phase1['education_system'] == system]
        row = {'Système Éducatif': system, 'Pays': ', '.join(system_data['country'].unique())}

        for metric in comparison_metrics:
            if metric in system_data.columns:
                mean_val = system_data[metric].mean()
                std_val = system_data[metric].std()
                row[f'{metric}_mean'] = round(mean_val, 2) if not pd.isna(mean_val) else None
                row[f'{metric}_std'] = round(std_val, 2) if not pd.isna(std_val) else 0

        system_comparison_data.append(row)

    df_system_analysis = pd.DataFrame(system_comparison_data)
    df_system_analysis.to_csv(output_dir + 'analyse_comparative_systemes.csv', index=False, encoding='utf-8')

    # Save additional results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir + 'phase1_resultats_finaux_simples.csv', index=False, encoding='utf-8')
    importance_df.to_csv(output_dir + 'importance_variables_simples.csv', index=False, encoding='utf-8')

    # ==================== FINAL SUMMARY ====================
    safe_print("\n" + "=" * 60)
    safe_print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    safe_print("=" * 60)

    safe_print("\n📊 PHASE 1 RESULTS:")
    safe_print(f"   • Systems analyzed: {len(system_classification)}")
    safe_print(f"   • Countries included: {len(df_phase1)}")
    safe_print(f"   • Clustering quality: {analyzer.results['clustering']['silhouette']:.3f}")
    safe_print(f"   • Classification accuracy: {sum([1 for r in results if r['correct']]) / len(results):.1%}")
    safe_print(f"   • Most important feature: {importance_df.iloc[0]['variable']}")

    safe_print("\n🤖 PHASE 2 RESULTS:")
    safe_print(f"   • AI impact data prepared for {len(df_phase2)} countries")

    safe_print("\n📈 VISUALIZATIONS GENERATED:")
    safe_print("   • PCA plots (Real systems vs Clusters)")
    safe_print("   • Feature importance analysis")
    safe_print("   • Classification results by country")
    safe_print("   • System comparison metrics")

    safe_print("\n📁 FILES GENERATED:")
    safe_print("   📊 phase1_analysis_report.html - Professional report")
    safe_print("   📈 comprehensive_visualization.png - PCA + Clusters + Features")
    safe_print("   📊 classification_and_comparison.png - Results + System comparison")
    safe_print("   📄 Multiple CSV datasets with processed data")
    safe_print("   📝 analysis.log - Execution log")

    safe_print("\n🔬 QUALITY ASSURANCE:")
    safe_print("   • UTF-8 encoding: Fixed throughout")
    safe_print("   • Error handling: Comprehensive logging")
    safe_print("   • Data validation: Missing value imputation")
    safe_print("   • Unit tests: Run 'python test_education_analyzer.py'")

    safe_print("\n📈 NEXT STEPS:")
    safe_print("   1. Open phase1_analysis_report.html in browser")
    safe_print("   2. Review PNG visualizations")
    safe_print("   3. Analyze AI impact correlations in Phase 2 data")
    safe_print("   4. Consider additional socioeconomic variables")

    safe_print("\n" + "=" * 60)

if __name__ == "__main__":
    main()