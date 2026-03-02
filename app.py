import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf # Import tensorflow explicitly for Adam optimizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration --- #
FILE_PATH = '/content/drive/MyDrive/Upi fraud dataset final.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
ANN_OPTIMAL_THRESHOLD = 0.22 # From previous analysis
XGB_OPTIMAL_THRESHOLD = 0.25 # From previous analysis
RISK_MEDIUM_THRESHOLD_PERCENT = 22 # Corresponds to ANN_OPTIMAL_THRESHOLD * 100
RISK_HIGH_THRESHOLD_PERCENT = 75 # Corresponds to 0.75 * 100

# --- Streamlit App Title --- #
st.title("UPI Fraud Detection Dashboard")
st.write("A comprehensive dashboard for fraud detection using ANN and XGBoost models, "
         "and a simple risk scoring system.")

# --- Data Loading and Preprocessing --- #
@st.cache_data
def load_and_preprocess_data():
    st.subheader("1. Data Loading and Preprocessing")
    st.info("Loading and preprocessing data... This may take a moment.")

    # Load the original UPI dataset
    upi_df = pd.read_csv(FILE_PATH)

    # Drop ID-like columns (uniqueness > 95%)
    columns_to_drop = []
    for col in upi_df.columns:
        unique_values_count = upi_df[col].nunique()
        total_rows = len(upi_df)
        uniqueness_percentage = (unique_values_count / total_rows) * 100
        if uniqueness_percentage > 95:
            columns_to_drop.append(col)
    if columns_to_drop:
        upi_df.drop(columns=columns_to_drop, inplace=True)
        st.write(f"Dropped ID-like columns: {columns_to_drop}")

    # One-hot encode categorical variables
    categorical_cols = upi_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        upi_df = pd.get_dummies(upi_df, columns=categorical_cols, drop_first=True)
        st.write(f"One-hot encoded categorical columns. New DataFrame shape: {upi_df.shape}")

    # Separate features (X) and target variable (y)
    X = upi_df.drop('fraud', axis=1)
    y = upi_df['fraud']

    # Perform stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Compute class weights
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, class_weights_array))
    st.write(f"Computed Class Weights: {class_weights}")

    # Scale features using StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames, preserving column names
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    st.success("Data loaded and preprocessed successfully!")
    st.write(f"Shape of X_train: {X_train.shape}")

    return X_train, X_test, y_train, y_test, class_weights

# --- ANN Model Building and Training --- #
@st.cache_resource
def build_and_train_ann_model(X_train, X_test, y_train, y_test, class_weights,
                              ann_dense_layer1_neurons, ann_dropout_rate1,
                              ann_dense_layer2_neurons, ann_dropout_rate2, ann_learning_rate, batch_size):
    st.subheader("2. ANN Model Training")
    st.info("Building and training the Artificial Neural Network model... This might take a few seconds.")

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(ann_dense_layer1_neurons, activation='relu'))
    model.add(Dropout(ann_dropout_rate1))
    model.add(Dense(ann_dense_layer2_neurons, activation='relu'))
    model.add(Dropout(ann_dropout_rate2))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=ann_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'recall', 'AUC'])

    early_stopping_callback = EarlyStopping(
        monitor='val_recall',
        patience=3,
        mode='max',
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping_callback],
        validation_data=(X_test, y_test),
        verbose=0
    )
    st.success("ANN Model trained successfully!")
    return model, history

# --- ANN Model Evaluation --- #
def evaluate_ann_model(model, X_test, y_test, initial_optimal_threshold, current_threshold=None):
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Determine which threshold to use for 'optimized' metrics
    threshold_for_metrics = current_threshold if current_threshold is not None else initial_optimal_threshold

    y_pred_thresholded = (y_pred_proba > threshold_for_metrics).astype(int)
    f1_optimized = f1_score(y_test, y_pred_thresholded, pos_label=1)
    recall_optimized = recall_score(y_test, y_pred_thresholded, pos_label=1)
    precision_optimized = precision_score(y_test, y_pred_thresholded, pos_label=1)

    conf_matrix = confusion_matrix(y_test, y_pred_thresholded)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    return {
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision, # default threshold
        'recall': recall, # default threshold
        'f1_score': f1, # default threshold
        'roc_auc': roc_auc,
        'f1_optimized': f1_optimized, # optimized/current threshold
        'recall_optimized': recall_optimized, # optimized/current threshold
        'precision_optimized': precision_optimized, # optimized/current threshold
        'conf_matrix': conf_matrix,
        'roc_curve_data': (fpr, tpr, roc_auc)
    }

# --- XGBoost Model Training and Evaluation --- #
@st.cache_resource
def build_and_train_xgb_model(X_train, y_train, class_weights,
                              xgb_n_estimators, xgb_max_depth, xgb_learning_rate):
    st.subheader("4. XGBoost Model Training")
    st.info("Building and training the XGBoost model...")

    # Correct scale_pos_weight calculation based on counts
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_corrected = neg_count / pos_count

    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        scale_pos_weight=scale_pos_weight_corrected,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    st.success("XGBoost Model trained successfully!")
    return xgb_model

@st.cache_data
def evaluate_xgb_model(_model, X_test, y_test, optimal_threshold):
    y_pred_proba_xgb = _model.predict_proba(X_test)[:, 1]

    # Threshold optimization for XGBoost (for finding the best F1, not necessarily using it below)
    thresholds = np.arange(0.2, 0.81, 0.01)
    best_f1_xgb = 0
    optimal_threshold_found_xgb = 0
    for threshold in thresholds:
        y_pred_current_xgb = (y_pred_proba_xgb > threshold).astype(int)
        f1_current_xgb = f1_score(y_test, y_pred_current_xgb, pos_label=1)
        if f1_current_xgb > best_f1_xgb:
            best_f1_xgb = f1_current_xgb
            optimal_threshold_found_xgb = threshold

    y_pred_optimized_xgb = (y_pred_proba_xgb > optimal_threshold).astype(int) # Use pre-defined optimal_threshold
    f1_optimized_xgb = f1_score(y_test, y_pred_optimized_xgb, pos_label=1)
    recall_optimized_xgb = recall_score(y_test, y_pred_optimized_xgb, pos_label=1)
    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

    return {
        'f1_optimized': f1_optimized_xgb,
        'recall_optimized': recall_optimized_xgb,
        'roc_auc': roc_auc_xgb,
        'optimal_threshold_found': optimal_threshold_found_xgb
    }


def streamlit_app():
    # Initialize session state variables if they don't exist
    if 'y_pred_proba_ann' not in st.session_state:
        st.session_state['y_pred_proba_ann'] = None
    if 'y_test_ann' not in st.session_state:
        st.session_state['y_test_ann'] = None
    if 'roc_curve_data_ann' not in st.session_state:
        st.session_state['roc_curve_data_ann'] = None
    if 'ann_model_trained' not in st.session_state:
        st.session_state['ann_model_trained'] = False

    # Load and preprocess data
    X_train_st, X_test_st, y_train_st, y_test_st, class_weights_st = load_and_preprocess_data()

    # --- Sidebar for Hyperparameter Controls --- #
    st.sidebar.header("Hyperparameter Controls")

    st.sidebar.subheader("ANN Model Parameters")
    ann_dense_layer1_neurons = st.sidebar.slider(
        'ANN Layer 1 Neurons', min_value=64, max_value=512, value=256, step=32
    )
    ann_dropout_rate1 = st.sidebar.slider(
        'ANN Dropout Rate 1', min_value=0.1, max_value=0.5, value=0.3, step=0.05
    )
    ann_dense_layer2_neurons = st.sidebar.slider(
        'ANN Layer 2 Neurons', min_value=32, max_value=256, value=128, step=16
    )
    ann_dropout_rate2 = st.sidebar.slider(
        'ANN Dropout Rate 2', min_value=0.1, max_value=0.5, value=0.3, step=0.05
    )
    ann_learning_rate = st.sidebar.slider(
        'ANN Learning Rate', min_value=0.0001, max_value=0.1, value=0.001, format="%.4f"
    )

    st.sidebar.subheader("XGBoost Model Parameters")
    xgb_n_estimators = st.sidebar.slider(
        'XGBoost n_estimators', min_value=50, max_value=200, value=80, step=10
    )
    xgb_max_depth = st.sidebar.slider(
        'XGBoost max_depth', min_value=3, max_value=10, value=6, step=1
    )
    xgb_learning_rate = st.sidebar.slider(
        'XGBoost Learning Rate', min_value=0.001, max_value=0.5, value=0.1, format="%.3f"
    )

    batch_size = st.sidebar.selectbox(
        'Batch Size', options=[128, 256, 512, 1024], index=2 # Default to 512
    )

    # --- Train Models Button --- #
    if st.sidebar.button('Train Models'):
        st.write("Training models with selected hyperparameters...")

        # Build and train the ANN model
        ann_model, ann_history = build_and_train_ann_model(
            X_train_st, X_test_st, y_train_st, y_test_st, class_weights_st,
            ann_dense_layer1_neurons, ann_dropout_rate1,
            ann_dense_layer2_neurons, ann_dropout_rate2, ann_learning_rate, batch_size
        )

        # Perform ANN evaluation (using the pre-defined optimal threshold for initial display)
        st.subheader("3. ANN Model Evaluation")
        st.info("Evaluating ANN model performance...")
        ann_eval_results = evaluate_ann_model(ann_model, X_test_st, y_test_st, ANN_OPTIMAL_THRESHOLD)
        st.success("ANN Model evaluation complete!")

        # Store results in session state for interactive section
        st.session_state['y_pred_proba_ann'] = ann_eval_results['y_pred_proba']
        st.session_state['y_test_ann'] = y_test_st
        st.session_state['roc_curve_data_ann'] = ann_eval_results['roc_curve_data']
        st.session_state['ann_model_trained'] = True

        st.write(f"**Optimal Threshold for ANN**: {ANN_OPTIMAL_THRESHOLD:.2f}")
        st.write(f"**Precision (Fraud) with Optimal Threshold**: {ann_eval_results['precision_optimized']:.4f}")
        st.write(f"**Recall (Fraud) with Optimal Threshold**: {ann_eval_results['recall_optimized']:.4f}")
        st.write(f"**F1-Score (Fraud) with Optimal Threshold**: {ann_eval_results['f1_optimized']:.4f}")
        st.write(f"**ROC-AUC**: {ann_eval_results['roc_auc']:.4f}")

        st.write("**Confusion Matrix (Optimal Threshold)**")
        st.write(ann_eval_results['conf_matrix'])

        st.write("**ROC Curve (with Optimal Threshold)**")
        fig_ann_roc, ax_ann_roc = plt.subplots(figsize=(8, 6))
        fpr_ann, tpr_ann, roc_auc_ann = ann_eval_results['roc_curve_data']
        ax_ann_roc.plot(fpr_ann, tpr_ann, color='blue', label=f'ROC curve (area = {roc_auc_ann:.2f})')

        # Plot the optimal threshold point on the ROC curve
        y_pred_thresholded = (ann_eval_results['y_pred_proba'] > ANN_OPTIMAL_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_st, y_pred_thresholded).ravel()
        fpr_at_opt_thresh = fp / (fp + tn)
        tpr_at_opt_thresh = tp / (tp + fn)
        ax_ann_roc.plot(fpr_at_opt_thresh, tpr_at_opt_thresh, 'o', color='green', markersize=8, label=f'Optimal Thresh ({ANN_OPTIMAL_THRESHOLD:.2f})')

        ax_ann_roc.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        ax_ann_roc.set_xlabel('False Positive Rate')
        ax_ann_roc.set_ylabel('True Positive Rate')
        ax_ann_roc.set_title('Receiver Operating Characteristic (ROC) Curve for ANN')
        ax_ann_roc.legend()
        ax_ann_roc.grid(True)
        st.pyplot(fig_ann_roc)

        # Build and train the XGBoost model
        xgb_model_st = build_and_train_xgb_model(
            X_train_st, y_train_st, class_weights_st,
            xgb_n_estimators, xgb_max_depth, xgb_learning_rate
        )

        st.subheader("5. XGBoost Model Evaluation") # Add subheader here
        st.info("Evaluating XGBoost model performance...") # Add info here
        xgb_eval_results = evaluate_xgb_model(xgb_model_st, X_test_st, y_test_st, XGB_OPTIMAL_THRESHOLD)
        st.success("XGBoost Model evaluation complete!") # Add success here

        st.write(f"**Optimal Threshold for XGBoost**: {XGB_OPTIMAL_THRESHOLD:.2f}")
        st.write(f"**F1-Score (Fraud) with Optimal Threshold**: {xgb_eval_results['f1_optimized']:.4f}")
        st.write(f"**Recall (Fraud) with Optimal Threshold**: {xgb_eval_results['recall_optimized']:.4f}")
        st.write(f"**ROC-AUC**: {xgb_eval_results['roc_auc']:.4f}")
        st.write(f"*(Optimal threshold for XGBoost found by iteration for best F1: {xgb_eval_results['optimal_threshold_found']:.2f})*")


        # --- Model Comparison --- #
        st.header("6. Model Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['F1-Score (Fraud)', 'Recall (Fraud)', 'ROC-AUC'],
            'ANN Model': [ann_eval_results['f1_optimized'], ann_eval_results['recall_optimized'], ann_eval_results['roc_auc']],
            'XGBoost Model': [xgb_eval_results['f1_optimized'], xgb_eval_results['recall_optimized'], xgb_eval_results['roc_auc']]
        })
        st.table(comparison_df)

        # --- Simple Risk Scoring (using ANN model's predictions) --- #
        st.header("7. Transaction Risk Scoring (using ANN model)")
        st.info("Converting ANN model's predicted probabilities into risk scores (0-100) and categorizing them.")

        # Convert predicted probabilities to risk scores (0-100)
        risk_scores = ann_eval_results['y_pred_proba'] * 100

        # Categorize risk scores
        risk_categories = []
        for score in risk_scores:
            if score < RISK_MEDIUM_THRESHOLD_PERCENT:
                risk_categories.append('Low Risk')
            elif score >= RISK_MEDIUM_THRESHOLD_PERCENT and score < RISK_HIGH_THRESHOLD_PERCENT:
                risk_categories.append('Medium Risk')
            else:
                risk_categories.append('High Risk')

        risk_categories_series = pd.Series(risk_categories, name='Risk Category')

        st.subheader("Distribution of Risk Categories (on Test Set)")
        st.write("**Number of transactions per category:**")
        st.write(risk_categories_series.value_counts())
        st.write("**Percentage of transactions per category:**")
        st.write(risk_categories_series.value_counts(normalize=True) * 100)

        st.markdown("--- ")
        st.write("This application provides an interactive dashboard for UPI fraud detection, "
                 "comparing ANN and XGBoost models, and demonstrating a simple risk scoring mechanism.")

    # --- Interactive Threshold Optimization (ANN Model) --- #
    st.header("8. Interactive Threshold Optimization (ANN Model)")
    if st.session_state['ann_model_trained'] and st.session_state['y_pred_proba_ann'] is not None:
        st.write("Adjust the slider to see how different classification thresholds affect the ANN model's performance metrics.")

        current_threshold = st.slider(
            'Select ANN Classification Threshold',
            min_value=0.0,
            max_value=1.0,
            value=ANN_OPTIMAL_THRESHOLD, # Default to the previously found optimal threshold
            step=0.01,
            key='ann_threshold_slider'
        )

        # Recalculate metrics based on the current slider threshold
        y_pred_proba_ann_st = st.session_state['y_pred_proba_ann']
        y_test_ann_st = st.session_state['y_test_ann']
        fpr_ann_st, tpr_ann_st, roc_auc_ann_st = st.session_state['roc_curve_data_ann']

        y_pred_dynamic = (y_pred_proba_ann_st > current_threshold).astype(int)

        precision_dynamic = precision_score(y_test_ann_st, y_pred_dynamic, pos_label=1)
        recall_dynamic = recall_score(y_test_ann_st, y_pred_dynamic, pos_label=1)
        f1_dynamic = f1_score(y_test_ann_st, y_pred_dynamic, pos_label=1)
        conf_matrix_dynamic = confusion_matrix(y_test_ann_st, y_pred_dynamic)

        st.write(f"**Metrics at Threshold {current_threshold:.2f}:**")
        st.write(f"Precision (Fraud): {precision_dynamic:.4f}")
        st.write(f"Recall (Fraud): {recall_dynamic:.4f}")
        st.write(f"F1-Score (Fraud): {f1_dynamic:.4f}")

        st.write("**Confusion Matrix:**")
        st.write(conf_matrix_dynamic)

        st.write("**ROC Curve with Current Threshold Point**")
        fig_ann_roc_dynamic, ax_ann_roc_dynamic = plt.subplots(figsize=(8, 6))
        ax_ann_roc_dynamic.plot(fpr_ann_st, tpr_ann_st, color='blue', label=f'ROC curve (area = {roc_auc_ann_st:.2f})')

        # Find the point on the ROC curve corresponding to the current_threshold
        # This requires calculating (FPR, TPR) for the specific threshold.
        tn, fp, fn, tp = confusion_matrix(y_test_ann_st, y_pred_dynamic).ravel()
        fpr_at_current_thresh = fp / (fp + tn)
        tpr_at_current_thresh = tp / (tp + fn)

        ax_ann_roc_dynamic.plot(fpr_at_current_thresh, tpr_at_current_thresh, 'o', color='green', markersize=8, label=f'Current Thresh ({current_threshold:.2f})')
        ax_ann_roc_dynamic.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        ax_ann_roc_dynamic.set_xlabel('False Positive Rate')
        ax_ann_roc_dynamic.set_ylabel('True Positive Rate')
        ax_ann_roc_dynamic.set_title('Receiver Operating Characteristic (ROC) Curve for ANN')
        ax_ann_roc_dynamic.legend()
        ax_ann_roc_dynamic.grid(True)
        st.pyplot(fig_ann_roc_dynamic)
    else:
        st.warning("Please train the models first to enable interactive threshold optimization.")

# Run the Streamlit app
streamlit_app()