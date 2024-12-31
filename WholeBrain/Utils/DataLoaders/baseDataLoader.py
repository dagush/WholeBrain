# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
import numpy as np

class DataLoader():
    def name(self):
        raise NotImplementedError('This should have been implemented by a subclass')

    def set_basePath(self, path):
        raise NotImplementedError('This should have been implemented by a subclass')

    def TR(self):
        raise NotImplementedError('This should have been implemented by a subclass')

    def N(self):
        raise NotImplementedError('This should have been implemented by a subclass')

    def get_groupLabels(self):
        raise NotImplementedError('This should have been implemented by a subclass')

    def get_classification(self):
        raise NotImplementedError('This should have been implemented by a subclass')

    def get_AvgSC_ctrl(self, normalized=True):
        raise NotImplementedError('This should have been implemented by a subclass')

    # -------------------------- Methods to gather extra information -------------------
    # -------------------------- (depends on each dataset) -----------------------------
    def get_SubjectData(self, subjectID):
        raise NotImplementedError('This should have been implemented by a subclass')

    def get_GroupData(self, group):
        raise NotImplementedError('This should have been implemented by a subclass')

    def get_GlobalData(self):
        return {
            'SC': self.get_AvgSC_ctrl()
        }

    # -------------------------- Convenience methods -----------------------------------
    # get_fullGroup_fMRI: convenience method to load all fMRIs for a given group
    def get_fullGroup_fMRI(self, group):
        data = self.get_fullGroup_data(group)
        fMRIs = {s: data[s]['timeseries'] for s in data}
        return fMRIs

    def get_groupSubjects(self, group):
        raise NotImplementedError('This should have been implemented by a subclass')

    def get_allStudySubjects(self):
        allStudySubjects = []
        for label in self.get_groupLabels():
            allStudySubjects += self.get_groupSubjects(label)
        return allStudySubjects

    def discardSubject(self, subjectID):
        raise NotImplementedError('This should have been implemented by a subclass')

    # -------------------------- Convenience methods -----------------------------------
    def get_groupLabels(self):  # Returns a list with all group labels
        classif = self.get_classification()
        labels = list({classif[s] for s in classif})
        return labels

    # get_fullGroup_data: convenience method to load all data for a given group
    def get_fullGroup_data(self, group):
        subjects = self.get_groupSubjects(group)
        data = {}
        for subject in subjects:
            data[subject] = self.get_subjectData(subject)[subject]
        return data

    def get_groupSubjects(self, group):  # return a list of all subjcets in the group
        classif = self.get_classification()
        data = []
        for subject in classif:
            if classif[subject] == group:
                data.append(subject)
        return data

    def get_allStudySubjects(self):
        allStudySubjects = []
        for label in self.get_groupLabels():
            allStudySubjects += self.get_groupSubjects(label)
        return allStudySubjects

    def discardSubject(self, subjectID):
        raise NotImplementedError('This should have been implemented by a subclass')

    def get_subjectDatum(self, subjectID, attribute):  # returns a single datum for a given subject
        data = self.get_subjectData(subjectID)
        if attribute in data[subjectID]:
            return data[subjectID][attribute]
        elif attribute == 'avgSC_ctrl':
            return self.get_AvgSC_ctrl()
        else:
            return None  # if the attribute does not exist in the subject data

    # ===================== compute the Avg SC matrix over the HC sbjects
    def __computeAvgSC_HC_Matrix(self, ctrl_label):
        HC = self.get_groupSubjects(ctrl_label)
        sumMatrix = np.zeros((self.N(), self.N()))
        for subject in HC:
            SC = self.get_subjectData(subject)[subject]['SC']
            sumMatrix += SC
        return sumMatrix / len(HC)  # but we normalize it afterwards, so we probably do not need this...

    # ===================== Normalize a SC matrix
    # Implements the two most basic methods, for any other one, should be overwritten...
    def _normalize_SC(self, SC,
                       normalizationMethod='maxSC',  # maxLogNode/maxSC
                       normalizationFactor=0.2,  # 0.7275543904602363
                       ):
        if normalizationMethod == 'maxLogNode':
            logMatrix = np.log(SC + 1)
            maxNodeInput = np.max(np.sum(logMatrix, axis=0))  # This is the same as np.max(logMatrix @ np.ones(N))
            finalMatrix = logMatrix * normalizationFactor / maxNodeInput
        elif normalizationMethod == 'maxSC':
            finalMatrix = SC / np.max(SC) * normalizationFactor
        else:
            finalMatrix = SC
        return finalMatrix

    # ===================== Compute an averaged SC matrix over the control group
    # Should be overwritten if an average matrix is provided...
    def get_AvgSC_ctrl(self, ctrl_label='HC', normalized='maxSC', normalizationFactor=0.2):  # returns a SINGLE SC matrix (average over control subjects)
        avgSC = self.__computeAvgSC_HC_Matrix(ctrl_label)
        normSC = self._normalize_SC(avgSC, normalizationMethod=normalized, normalizationFactor=normalizationFactor)
        return normSC

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF