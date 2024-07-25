# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
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
        raise NotImplementedError('This should have been implemented by a subclass')

    def get_groupSubjects(self, group):
        raise NotImplementedError('This should have been implemented by a subclass')

    def get_allStudySubjects(self):
        allStudySubjects = []
        for label in self.get_groupLabels():
            allStudySubjects += self.get_groupSubjects(label)
        return allStudySubjects

    def discardSubject(self, subjectID):
        raise NotImplementedError('This should have been implemented by a subclass')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF