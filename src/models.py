
from data.dataset import X_train, X_test, X_val, y_train, y_test, y_val
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report as cr
import mlflow
from mlflow.tracking import MlflowClient
from src.utils import gen_mlflow_metrics

class models():
    def __init__(self, params):
        self.bests_dic, self.report_d, self.rn = None, None, 1
        self.params, self.map = params[0], params[1]

    def BestModels(self, verbose=True):
        self.bests_dic = {}
        for i in self.map:
            if i in self.params:
                Gridsearch = GridSearchCV(self.map[i], self.params[i], cv=5)
                Gridsearch.fit(X_train, y_train)
                best_model = Gridsearch.best_estimator_
            else:
                best_model = self.map[i]
                best_model.fit(X_train, y_train)
            self.bests_dic.update({i: best_model})
            if verbose == True:
                print(f'\n{i}: {best_model}')

    def Val_reports(self, verbose=True):
        self.report_d = {}
        if self.bests_dic is None:
            self.BestModels(verbose=False)
        for i in self.bests_dic:
            y_val_pred = self.bests_dic[i].predict(X_val)
            report, reportd = cr(y_val, y_val_pred), cr(y_val, y_val_pred, output_dict=True)
            self.report_d.update({i:reportd})
            if verbose == True:
                print(f'\n{i}: \n{report}')

    def track(self):
        '''MLFLOW TRACKING'''
        if self.report_d == None:
            self.Val_reports(verbose=False)
        client, exp_name, uri = MlflowClient(), str(input('Experiment Name: ')), str(input('Experiment URI: '))
        #mlflow.set_experiment(exp_name)
        mlflow.set_tracking_uri(uri)
        experiment = client.get_experiment_by_name(exp_name)
        if experiment is None:
            experiment_id = client.create_experiment(exp_name)
            experiment = client.get_experiment(experiment_id)
        mlflow.set_experiment(exp_name)
        for i in self.report_d:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id], 
                                        filter_string=f"tag.mlflow.runName = '{i}'")
            if len(runs) == 0:
                with mlflow.start_run(run_name=i): 
                    if i in self.params:
                        mlflow.log_params(self.params[i])
                    mlflow_metrics = gen_mlflow_metrics(self.report_d[i])
                    mlflow.log_metrics(mlflow_metrics)
                    mlflow.sklearn.log_model(self.bests_dic[i], name=i,
                                            input_example=X_val.iloc[[5]])
            else:
                print(f"Experiment {exp_name} already contains a run '{i}' !!")

    def Auto(self, met='accuracy', avg='weighted avg'):
        auto = []
        if self.report_d is None:
            self.Val_reports(verbose=False)
        if met == 'accuracy':
            for i in self.report_d:
                if len(auto) == 0 or self.report_d[i][met]>auto[-1][1]:
                    auto.append([i, self.report_d[i][met]])
            return self.bests_dic[auto[-1][0]]
        elif met in ['precision', 'recall', 'f1-score']:
            for i in self.report_d:
                if len(auto) == 0 or self.report_d[i][avg][met]>auto[-1][1]:
                    if avg == 'macro avg':
                        auto.append([i, self.report_d[i]['macro avg'][met]])
                    elif avg == 'weighted avg':
                        auto.append([i, self.report_d[i]['weighted avg'][met]])
            return self.bests_dic[auto[-1][0]]
        else:
            print("Please select one of 'accuracy', 'precision', 'recall', 'f1-score'")

    def Use(self, mod_choice):
        if self.bests_dic is None:
            self.BestModels(verbose=False)
        if mod_choice in self.bests_dic:
            return self.bests_dic[mod_choice]
        else:
            print(f'Please choose one of {list(self.bests_dic.keys())}')