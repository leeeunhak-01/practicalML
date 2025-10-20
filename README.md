# 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 데이터셋 생성, 분할, 모델링, 평가 관련 라이브러리
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# imbalanced-learn에서 샘플링 기법 임포트
from imblearn.under_sampling import TomekLinks, CondensedNearestNeighbour, OneSidedSelection
from imblearn.over_sampling import SMOTE, ADASYN

# 결정 경계 및 데이터 분포를 시각화하는 함수
def plot_decision_boundary(X, y, model, title):
    """
    Visualizes the decision boundary for a 2D dataset and a trained model.
    Args:
        X (np.array): Feature data
        y (np.array): Label data
        model: Trained classification model
        title (str): Title for the plot
    """
    # 각 클래스에 대해 다른 색상으로 산점도 생성
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], alpha=0.6, label='Majority Class (Class 0)', s=25)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], alpha=0.8, label='Minority Class (Class 1)', s=25)

    # 결정 경계를 그리기 위한 그리드 생성
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 150),
                         np.linspace(ylim[0], ylim[1], 150))
    
    # 그리드의 각 포인트에 대해 모델 예측 수행
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 등고선 플롯을 사용하여 결정 경계 표시
    ax.contourf(xx, yy, Z, alpha=0.2)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 1. 불균형 데이터셋 생성
# 1000개의 샘플, 2개의 특성, 95:5 클래스 비율을 가진 데이터셋
# class_sep 값을 조정하여 클래스 간의 거리를 좁힘 (언더샘플링 효과를 보기 위함)
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1, weights=[0.95, 0.05],
                           flip_y=0, random_state=42, class_sep=0.8)

print(f"Original dataset class distribution: {Counter(y)}")

# 데이터를 학습용과 테스트용으로 분할 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training dataset class distribution: {Counter(y_train)}")
print(f"Test dataset class distribution: {Counter(y_test)}")
print("-" * 50)

# 2. 샘플링 기법 정의
# 적용할 언더샘플링 및 오버샘플링 기법 딕셔너리
samplers = {
    "Original Data": None,
    "Tomek Links (Undersampling)": TomekLinks(sampling_strategy='auto'),
    "CNN (Undersampling)": CondensedNearestNeighbour(sampling_strategy='auto', n_neighbors=3, random_state=42),
    "One-Sided Selection (Undersampling)": OneSidedSelection(n_neighbors=3, random_state=42),
    "SMOTE (Oversampling)": SMOTE(sampling_strategy='auto', random_state=42),
    "ADASYN (Oversampling)": ADASYN(sampling_strategy='auto', random_state=42)
}

# 결과 시각화를 위한 Figure 설정 (6개 플롯에 맞게 크기 조정)
plt.figure(figsize=(22, 14))

# 최종 성능 평가 결과를 저장할 딕셔너리
final_reports = {}

# 3 & 4. 각 샘플링 기법 적용, 모델 학습, 평가 및 시각화
for i, (name, sampler) in enumerate(samplers.items()):
    ax = plt.subplot(2, 3, i + 1)
    
    X_resampled, y_resampled = X_train, y_train
    
    # 샘플링 적용 ('Original Data'는 건너뜀)
    if sampler:
        print(f"Applying [{name}] sampling...")
        # CNN은 시간이 오래 걸릴 수 있음
        if name == "CNN (Undersampling)":
            print("Note: CondensedNearestNeighbour can be slower than other methods.")
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f"Class distribution after sampling: {Counter(y_resampled)}")
        print("-" * 50)
        
    # 로지스틱 회귀 모델 학습
    model = LogisticRegression(solver='lbfgs', random_state=42)
    model.fit(X_resampled, y_resampled)
    
    # 테스트 데이터로 예측 및 성능 평가
    y_pred = model.predict(X_test)
    final_reports[name] = classification_report(y_test, y_pred, digits=4)
    
    # 시각화
    plot_title = f"{name}\nDistribution: {Counter(y_resampled)}"
    plot_decision_boundary(X_resampled, y_resampled, model, plot_title)

# 전체 레이아웃 조정 및 플롯 표시
plt.tight_layout(pad=3.0)
plt.show()

# 5. 최종 성능 평가 결과 요약 출력
print("\n" + "=" * 60)
print("Final Model Performance Comparison by Sampling Method")
print("=" * 60)
for name, report in final_reports.items():
    print(f"\n--- {name} Model Performance ---")
    print(report)

