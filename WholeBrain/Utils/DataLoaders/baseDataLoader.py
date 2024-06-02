# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class DataLoader():
    def set_basePath(self, path):
        raise Exception('This should have been implemented by a subclass')

    def get_groupLabels(self):
        raise Exception('This should have been implemented by a subclass')

    def get_classification(self):
        raise Exception('This should have been implemented by a subclass')

    def get_SubjectData(self, subjectID):
        raise Exception('This should have been implemented by a subclass')

    def get_AvgSC_ctrl(self, normalized=True):
        raise Exception('This should have been implemented by a subclass')

    # -------------------------- Convenience methods -----------------------------------
    # get_fullGroup_fMRI: convenience method to load all fMRIs for a given subject group
    def get_fullGroup_fMRI(self, group):
        raise Exception('This should have been implemented by a subclass')

    def get_groupSubjects(self, group):
        raise Exception('This should have been implemented by a subclass')

    def get_allStudySubjects(self):
        allStudySubjects = []
        for label in self.get_groupLabels():
            allStudySubjects += self.get_groupSubjects(label)
        return allStudySubjects

    def discardSubject(self, subjectID):
        raise Exception('This should have been implemented by a subclass')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF