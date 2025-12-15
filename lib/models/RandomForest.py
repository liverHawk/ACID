"""
RandomForestモジュール（最適化余地なし、そのまま実装）
"""
from sklearn.ensemble import RandomForestClassifier
from typing import Any


class RandomForest(object):
    """
    RandomForest分類器のラッパークラス
    sklearnのラッパーで最適化の余地は少ない
    """

    def __init__(self, n_estimators: int = 200):
        super(RandomForest, self).__init__()
        self.cls = RandomForestClassifier(n_estimators=n_estimators)

    def fit(self, X: Any, y: Any) -> None:
        """モデルを訓練"""
        self.cls.fit(X, y)

    def predict(self, x: Any) -> Any:
        """予測を実行"""
        return self.cls.predict(x)
