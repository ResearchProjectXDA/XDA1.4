import sys
import os
import glob
import time
import warnings
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from CustomPlanner import CustomPlanner
from AnchorsPlanner import AnchorsPlanner
from model.ModelConstructor import constructModel
import explainability_techniques.LIME as lime
from NSGA3Planner import NSGA3Planner
from SHAPCustomPlanner import SHAPCustomPlanner
from FICustomPlanner import FICustomPlanner
from util import vecPredictProba, evaluateAdaptations
from FITEST import FitestPlanner
from RandomCustomPlanner import RandomPlanner


# import multilabel_oversampling as mo


# success score function (based on the signed distance with respect to the target success probabilities)
def successScore(adaptation, reqClassifiers, targetSuccessProba):
    return np.sum(vecPredictProba(reqClassifiers, [adaptation])[0] - targetSuccessProba)


def normalizeAdaptation(adaptation):
    new_adaptation = []
    for index in range(n_controllableFeatures):
        new_adaptation.append(((adaptation[index] - controllableFeatureDomains[index][0]) / (
                    controllableFeatureDomains[index][1] - controllableFeatureDomains[index][0])) * 100)

    return new_adaptation


# provided optimization score function (based on the ideal controllable feature assignment)
def optimizationScore(adaptation):
    adaptation = normalizeAdaptation(adaptation)
    score = 0
    tot = 100 * n_controllableFeatures
    for i in range(n_controllableFeatures):
        if optimizationDirections[i] == 1:
            score += 100 - adaptation[i]
        else:
            score += adaptation[i]
    score = score / tot
    return 1 - score

# ====================================================================================================== #
# IMPORTANT: everything named as custom in the code refers to the XDA approach                           #
#            everything named as confidence in the code refers to the predicted probabilities of success #
# ====================================================================================================== #


if __name__ == '__main__':
    programStartTime = time.time()

    os.chdir(sys.path[0])

    # suppress all warnings
    warnings.filterwarnings("ignore")

    # evaluate adaptations
    evaluate = True

    ds = pd.read_csv('../datasets/dataset5000.csv')
    # featureNames = ['formation', 'flying_speed', 'countermeasure', 'weather', 'day_time', 'threat_range', '#threats'] #uav
    featureNames = ['cruise speed','image resolution','illuminance','controls responsiveness','power',
     'smoke intensity','obstacle size','obstacle distance','firm obstacle'] #robot
    #featureNames = ['car_speed','p_x','p_y','orientation','weather','road_shape'] #drive
    controllableFeaturesNames = featureNames[0:3]
    externalFeaturesNames = featureNames[3:7]
    controllableFeatureIndices = [0, 1, 2]

    # for simplicity, we consider all the ideal points to be 0 or 100
    # so that we just need to consider ideal directions instead
    # -1 => minimize, 1 => maximize
    optimizationDirections = [1, 1, -1]

    #reqs = ["req_0", "req_1", "req_2"] #drive
    reqs = ["req_0", "req_1", "req_2", "req_3"] #robot
    # reqs = ["req_0", "req_1", "req_2", "req_3", "req_4",
            # "req_5", "req_6", "req_7", "req_8", "req_9", "req_10", "req_11"] #uav
    n_reqs = len(reqs)
    n_neighbors = 10
    n_startingSolutions = 10
    n_controllableFeatures = len(controllableFeaturesNames)

    targetConfidence = np.full((1, n_reqs), 0.8)[0]

    # split the dataset
    X = ds.loc[:, featureNames]
    y = ds.loc[:, reqs]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = []
    for req in reqs:
        print(Fore.RED + "Requirement: " + req + "\n" + Style.RESET_ALL)
        models.append(constructModel(X_train.values,
                                     X_test.values,
                                     np.ravel(y_train.loc[:, req]),
                                     np.ravel(y_test.loc[:, req])))
        print("=" * 100)

    controllableFeatureDomains = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]])
    discreteIndices = [0, 2]
    # initialize planners

    ds_train = pd.DataFrame(np.hstack((X_train, y_train)), columns=featureNames + reqs)
    trainPath = "../datasets/X_train.csv"
    ds_train.to_csv(trainPath, index=False)
    anchorsPlanner = AnchorsPlanner(trainPath, models, reqs, 0.95, len(featureNames),featureNames,controllableFeatureIndices, controllableFeatureDomains)

    customPlanner = CustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
                                  controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
                                  optimizationDirections, optimizationScore, 1, "../explainability_plots")

    nsga3Planner = NSGA3Planner(models, targetConfidence, controllableFeatureIndices, controllableFeatureDomains,
                                optimizationDirections, successScore, optimizationScore)

    # SHAPcustomPlanner = SHAPCustomPlanner(X_train, n_neighbors, n_startingSolutions, models, targetConfidence,
    #                                       controllableFeaturesNames, controllableFeatureIndices,
    #                                       controllableFeatureDomains,
    #                                       optimizationDirections, optimizationScore, 1, "../explainability_plots")

    # FICustomPlanner = FICustomPlanner(X_train, y_train, n_neighbors, n_startingSolutions, models, targetConfidence,
    #                                   controllableFeaturesNames, controllableFeatureIndices, controllableFeatureDomains,
    #                                   optimizationDirections, optimizationScore, 1, "../explainability_plots")

    # pop_size = nsga3Planner.algorithm.pop_size

    # FitestPlanner = FitestPlanner(models, targetConfidence,
    #                               controllableFeatureIndices, controllableFeatureDomains, optimizationScore,
    #                               successScore,
    #                               pop_size,
    #                               discreteIndices, 4, [0.8, 0.8, 0.8,0.8])

    # RandomPlanner = RandomPlanner(controllableFeatureIndices, controllableFeatureDomains, discreteIndices, models,
    #                               optimizationScore)

    # create lime explainer
    limeExplainer = lime.createLimeExplainer(X_train)

    # metrics
    # NOT  USED
    meanCustomScore = 0
    meanCustomScoreSHAP = 0
    meanCustomScoreFI = 0
    meanCustomScoreFitest = 0
    meanCustomScoreRandom = 0
    meanSpeedupSHAP = 0
    meanSpeedupFI = 0
    meanSpeedupFitest = 0
    meanSpeedupRandom = 0
    meanScoreDiffSHAP = 0
    meanScoreDiffFI = 0
    meanScoreDiffFitest = 0
    meanScoreDiffRandom = 0
    failedAdaptations = 0
    failedAdaptationsSHAP = 0
    failedAdaptationsFI = 0
    failedAdaptationsFitest = 0
    failedAdaptationsRandom = 0

    # adaptations
    results = []
    resultsAnchors = []
    resultsSHAP = []
    resultsFI = []
    resultsFitest = []
    resultsRandom = []
    customDataset = []
    nsga3Dataset = []
    resultsNSGA = []

    path = "../explainability_plots/adaptations"
    if not os.path.exists(path):
        os.makedirs(path)

    files = glob.glob(path + "/*")
    for f in files:
        os.remove(f)

    testNum = 10
    outputs_anchors = np.zeros((testNum, n_reqs))
    outputs_PDP = np.zeros((testNum, n_reqs))
    outputs_NSGA = np.zeros((testNum, n_reqs))
    for k in range(1, testNum + 1):
        rowIndex = k - 1
        row = X_test.iloc[rowIndex, :].to_numpy()

        print(Fore.BLUE + "Test " + str(k) + ":" + Style.RESET_ALL)
        print("Row " + str(rowIndex) + ":\n" + str(row))
        print("-" * 100)

        for i, req in enumerate(reqs):
            lime.saveExplanation(lime.explain(limeExplainer, models[i], row),
                                 path + "/" + str(k) + "_" + req + "_starting")
        # Anchors adaptation test
        startTime = time.time()
        customAdaptation, customConfidence, outputs = anchorsPlanner.evaluate_sample(row)
        outputs_anchors[rowIndex, :] = customConfidence
        endTime = time.time()
        customTime = endTime - startTime

        if customAdaptation is not None:
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], customAdaptation),
                                     path + "/" + str(k) + "_" + req + "_final")

            print("Best adaptation Anchors:                 " + str(customAdaptation[0:n_controllableFeatures]))
            print("Model confidence:                " + str(customConfidence))
            #print("Adaptation score:                " + str(customScore) + " /" + str(1))
        else:
            print("No adaptation found")
            customScore = None

        print("Anchors algorithm execution time: " + str(customTime) + " s")
        print("-" * 100)

        # customPlanner adaptation test
        startTime = time.time()
        customAdaptation, customConfidence, customScore = customPlanner.findAdaptation(row)
        outputs_PDP[rowIndex, :] = customConfidence
        endTime = time.time()
        customTime = endTime - startTime

        if customAdaptation is not None:
            for i, req in enumerate(reqs):
                lime.saveExplanation(lime.explain(limeExplainer, models[i], customAdaptation),
                                     path + "/" + str(k) + "_" + req + "_final")

            print("Best adaptation:                 " + str(customAdaptation[0:n_controllableFeatures]))
            print("Model confidence:                " + str(customConfidence))
            print("Adaptation score:                " + str(customScore) + " /" + str(1))
        else:
            print("No adaptation found")
            customScore = None

        print("Custom algorithm execution time: " + str(customTime) + " s")
        print("-" * 100)

        # SHAP adaptation test

        # startTime = time.time()
        # SHAPcustomAdaptation, SHAPcustomConfidence, SHAPcustomScore = SHAPcustomPlanner.findAdaptation(row)
        # endTime = time.time()
        # SHAPcustomTime = endTime - startTime

        # if SHAPcustomAdaptation is not None:
        #     for i, req in enumerate(reqs):
        #         lime.saveExplanation(lime.explain(limeExplainer, models[i], SHAPcustomAdaptation),
        #                              path + "/" + str(k) + "_" + req + "_final")

        #     print("Best adaptation SHAP:                 " + str(SHAPcustomAdaptation[0:n_controllableFeatures]))
        #     print("Model confidence SHAP:                " + str(SHAPcustomConfidence))
        #     print("Adaptation score SHAP:                " + str(SHAPcustomScore) + " /" + str(1))
        # else:
        #     print("No adaptation found")
        #     SHAPcustomScore = None

        # print("Custom SHAP algorithm execution time: " + str(SHAPcustomTime) + " s")
        # print("-" * 100)

        # Feature Importance custom adaptation test
        # startTime = time.time()
        # FIcustomAdaptation, FIcustomConfidence, FIcustomScore = FICustomPlanner.findAdaptation(row)
        # endTime = time.time()
        # FIcustomTime = endTime - startTime

        # if FIcustomAdaptation is not None:
        #     for i, req in enumerate(reqs):
        #         lime.saveExplanation(lime.explain(limeExplainer, models[i], FIcustomAdaptation),
        #                              path + "/" + str(k) + "_" + req + "_final")

        #     print("Best adaptation FI:                 " + str(FIcustomAdaptation[0:n_controllableFeatures]))
        #     print("Model confidence FI:                " + str(FIcustomConfidence))
        #     print("Adaptation score FI:                " + str(FIcustomScore) + " /" + str(1))
        # else:
        #     print("No adaptation found")
        #     FIcustomScore = None

        # print("Custom FI algorithm execution time: " + str(FIcustomTime) + " s")
        # print("-" * 100)

        # Fitest adaptation test
        # startTime = time.time()
        # FitestcustomAdaptation, FitestcustomConfidence, FitestcustomScore = FitestPlanner.run_search(row)
        # endTime = time.time()
        # FitestcustomTime = endTime - startTime

        # if FitestcustomAdaptation is not None:
        #     for i, req in enumerate(reqs):
        #         lime.saveExplanation(lime.explain(limeExplainer, models[i], FitestcustomAdaptation),
        #                              path + "/" + str(k) + "_" + req + "_final")

        #     print("Best adaptation Fitest:                 " + str(FitestcustomAdaptation[0:n_controllableFeatures]))
        #     print("Model confidence Fitest:                " + str(FitestcustomConfidence))
        #     print("Adaptation score Fitest:                " + str(FitestcustomScore) + " /" + str(1))
        # else:
        #     print("No adaptation found")
        #     FitestcustomScore = None

        # print("Fitest algorithm execution time: " + str(FitestcustomTime) + " s")
        # print("-" * 100)

        # Random adaptation test

        # startTime = time.time()
        # RandomCustomAdaptation, RandomCustomConfidence, RandomCustomScore = RandomPlanner.findAdaptation(row)
        # endTime = time.time()
        # RandomcustomTime = endTime - startTime

        # if RandomCustomAdaptation is not None:
        #     for i, req in enumerate(reqs):
        #         lime.saveExplanation(lime.explain(limeExplainer, models[i], RandomCustomAdaptation),
        #                              path + "/" + str(k) + "_" + req + "_final")

        #     print("Best adaptation Random:                 " + str(RandomCustomAdaptation[0:n_controllableFeatures]))
        #     print("Model confidence Random:                " + str(RandomCustomConfidence))
        #     print("Adaptation score Random:                " + str(RandomCustomScore) + " /" + str(1))
        # else:
        #     print("No adaptation found")
        #     RandomCustomScore = None

        # print("Custom Random algorithm execution time: " + str(RandomcustomTime) + " s")
        # print("-" * 100)

        externalFeatures = row[n_controllableFeatures:]

        # NSGA3 adaptation test
        startTime = time.time()
        nsga3Adaptation, nsga3Confidence, nsga3Score = nsga3Planner.findAdaptation(externalFeatures)
        endTime = time.time()
        nsga3Time = endTime - startTime
        outputs_NSGA[rowIndex, :] = nsga3Confidence

        print("Best NSGA3 adaptation:           " + str(nsga3Adaptation[:n_controllableFeatures]))
        print("Model confidence:                " + str(nsga3Confidence))
        print("Adaptation score:                " + str(nsga3Score) + " /" + str(1))
        print("NSGA3 execution time:            " + str(nsga3Time) + " s")

        print("-" * 100)
        
        resultsAnchors.append([customAdaptation,
                               customConfidence,
                               customScore,
                               customTime])
        
        results.append([customAdaptation,
                        customConfidence,
                        customScore,
                        customTime])

        resultsNSGA.append([nsga3Adaptation,
                            nsga3Confidence,
                            nsga3Score,
                         nsga3Time])

        # resultsSHAP.append([SHAPcustomAdaptation,
        #                     SHAPcustomConfidence,
        #                     SHAPcustomScore,
        #                     SHAPcustomTime])

        # resultsFI.append([FIcustomAdaptation,
        #                   FIcustomConfidence,
        #                   FIcustomScore,
        #                   FIcustomTime])

        # resultsFitest.append([FitestcustomAdaptation,
        #                       FitestcustomConfidence,
        #                       FitestcustomScore,
        #                       FitestcustomTime])

        # resultsRandom.append([RandomCustomAdaptation,
        #                       RandomCustomConfidence,
        #                       RandomCustomScore,
        #                       RandomcustomTime])
    
    #Metrics
    num_misclassified_req0_anchors = np.where(outputs_anchors[:, 0] < 0.5)[0].shape[0]
    num_misclassified_req1_anchors = np.where(outputs_anchors[:, 1] < 0.5)[0].shape[0]
    num_misclassified_req2_anchors = np.where(outputs_anchors[:, 2] < 0.5)[0].shape[0]
    num_misclassified_req3_anchors = np.where(outputs_anchors[:, 3] < 0.5)[0].shape[0]

    num_misclassified_req0_custom = np.where(outputs_PDP[:, 0] < 0.5)[0].shape[0]
    num_misclassified_req1_custom = np.where(outputs_PDP[:, 1] < 0.5)[0].shape[0]
    num_misclassified_req2_custom = np.where(outputs_PDP[:, 2] < 0.5)[0].shape[0]
    num_misclassified_req3_custom = np.where(outputs_PDP[:, 3] < 0.5)[0].shape[0]

    num_misclassified_req0_nsga = np.where(outputs_NSGA[:, 0] < 0.5)[0].shape[0]
    num_misclassified_req1_nsga = np.where(outputs_NSGA[:, 1] < 0.5)[0].shape[0]
    num_misclassified_req2_nsga = np.where(outputs_NSGA[:, 2] < 0.5)[0].shape[0]
    num_misclassified_req3_nsga = np.where(outputs_NSGA[:, 3] < 0.5)[0].shape[0]

    print(Fore.GREEN + "Number of misclassified req0 Anchors: " + str(num_misclassified_req0_anchors) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req1 Anchors: " + str(num_misclassified_req1_anchors) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req2 Anchors: " + str(num_misclassified_req2_anchors) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req3 Anchors: " + str(num_misclassified_req3_anchors) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req0 Custom: " + str(num_misclassified_req0_custom) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req1 Custom: " + str(num_misclassified_req1_custom) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req2 Custom: " + str(num_misclassified_req2_custom) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req3 Custom: " + str(num_misclassified_req3_custom) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req0 NSGA3: " + str(num_misclassified_req0_nsga) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req1 NSGA3: " + str(num_misclassified_req1_nsga) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req2 NSGA3: " + str(num_misclassified_req2_nsga) + Style.RESET_ALL)
    print(Fore.GREEN + "Number of misclassified req3 NSGA3: " + str(num_misclassified_req3_nsga) + Style.RESET_ALL)

    #Cross- entrhopy
    cross_entropy_req0_anchors = -np.sum(np.log(outputs_anchors[:,0]))
    cross_entropy_req1_anchors = -np.sum(np.log(outputs_anchors[:,1]))
    cross_entropy_req2_anchors = -np.sum(np.log(outputs_anchors[:,2]))
    cross_entropy_req3_anchors = -np.sum(np.log(outputs_anchors[:,3]))

    cross_entropy_req0_custom = -np.sum(np.log(outputs_PDP[:,0]))
    cross_entropy_req1_custom = -np.sum(np.log(outputs_PDP[:,1]))
    cross_entropy_req2_custom = -np.sum(np.log(outputs_PDP[:,2]))
    cross_entropy_req3_custom = -np.sum(np.log(outputs_PDP[:,3]))

    cross_entropy_req0_nsga = -np.sum(np.log(outputs_NSGA[:,0]))
    cross_entropy_req1_nsga = -np.sum(np.log(outputs_NSGA[:,1]))
    cross_entropy_req2_nsga = -np.sum(np.log(outputs_NSGA[:,2]))
    cross_entropy_req3_nsga = -np.sum(np.log(outputs_NSGA[:,3]))

    print(Fore.GREEN + "Cross-entropy req0 Anchors: " + str(cross_entropy_req0_anchors) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req1 Anchors: " + str(cross_entropy_req1_anchors) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req2 Anchors: " + str(cross_entropy_req2_anchors) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req3 Anchors: " + str(cross_entropy_req3_anchors) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req0 Custom: " + str(cross_entropy_req0_custom) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req1 Custom: " + str(cross_entropy_req1_custom) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req2 Custom: " + str(cross_entropy_req2_custom) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req3 Custom: " + str(cross_entropy_req3_custom) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req0 NSGA3: " + str(cross_entropy_req0_nsga) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req1 NSGA3: " + str(cross_entropy_req1_nsga) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req2 NSGA3: " + str(cross_entropy_req2_nsga) + Style.RESET_ALL)
    print(Fore.GREEN + "Cross-entropy req3 NSGA3: " + str(cross_entropy_req3_nsga) + Style.RESET_ALL)

    resultsAnchors = pd.DataFrame(resultsAnchors, columns=["custom_adaptation",
                                                           "custom_confidence",
                                                           "custom_score",
                                                           "custom_time"])
    
    results = pd.DataFrame(results, columns=["custom_adaptation",
                                             "custom_confidence",
                                             "custom_score",
                                             "custom_time"])

    resultsNSGA = pd.DataFrame(resultsNSGA, columns=["custom_adaptation",
                                                     "custom_confidence",
                                                     "custom_score",
                                                     "custom_time"])

    # resultsSHAP = pd.DataFrame(resultsSHAP, columns=["custom_adaptation",
    #                                                  "custom_confidence",
    #                                                  "custom_score",
    #                                                  "custom_time"])

    # resultsFI = pd.DataFrame(resultsFI, columns=["custom_adaptation",
    #                                              "custom_confidence",
    #                                              "custom_score",
    #                                              "custom_time"])

    # resultsFitest = pd.DataFrame(resultsFitest, columns=["custom_adaptation",
    #                                                      "custom_confidence",
    #                                                      "custom_score",
    #                                                      "custom_time"])

    # resultsRandom = pd.DataFrame(resultsRandom, columns=["custom_adaptation",
    #                                                      "custom_confidence",
    #                                                      "custom_score",
    #                                                      "custom_time"])

    path = "../results"
    if not os.path.exists(path):
        os.makedirs(path)

    resultsAnchors.to_csv(path + "/resultsAnchors.csv")
    results.to_csv(path + "/results.csv")
    # resultsSHAP.to_csv(path + "/resultsSHAP.csv")
    # resultsFI.to_csv(path + "/resultsFI.csv")
    # resultsFitest.to_csv(path + "/resultsFitest.csv")
    # resultsRandom.to_csv(path + "/resultsRandom.csv")
    resultsNSGA.to_csv(path + "/resultsNSGA.csv")

    if evaluate:
        # evaluateAdaptations(resultsAnchors, results, resultsSHAP, resultsFI, resultsFitest, resultsRandom, resultsNSGA, featureNames)
        evaluateAdaptations(resultsAnchors, results, resultsNSGA, featureNames)
    programEndTime = time.time()
    totalExecutionTime = programEndTime - programStartTime
    print("\nProgram execution time: " + str(totalExecutionTime / 60) + " m")
