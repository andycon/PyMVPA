#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Cross-validate a classifier on a dataset"""

__docformat__ = 'restructuredtext'

from mvpa.misc.copy import copy

from mvpa.measures.base import DatasetMeasure
from mvpa.datasets.splitter import NoneSplitter
from mvpa.base import warning
from mvpa.misc.state import StateVariable, Harvestable
from mvpa.misc.transformers import GrandMean

if __debug__:
    from mvpa.base import debug


class CrossValidatedTransferError(DatasetMeasure, Harvestable):
    """Cross validate a classifier on datasets generated by a splitter from a
    source dataset.

    Arbitrary performance/error values can be computed by specifying an error
    function (used to compute an error value for each cross-validation fold)
    and a combiner function that aggregates all computed error values across
    cross-validation folds.
    """

    results = StateVariable(enabled=False, doc=
       """Store individual results in the state""")
    splits = StateVariable(enabled=False, doc=
       """Store the actual splits of the data. Can be memory expensive""")
    transerrors = StateVariable(enabled=False, doc=
       """Store copies of transerrors at each step""")
    confusion = StateVariable(enabled=False, doc=
       """Store total confusion matrix (if available)""")
    training_confusion = StateVariable(enabled=False, doc=
       """Store total training confusion matrix (if available)""")
    samples_error = StateVariable(enabled=False,
                        doc="Per sample errors.")


    def __init__(self,
                 transerror,
                 splitter=NoneSplitter(),
                 combiner=GrandMean,
                 harvest_attribs=None,
                 copy_attribs='copy',
                 **kwargs):
        """
        Cheap initialization.

        :Parameters:
            transerror : TransferError instance
                Provides the classifier used for cross-validation.
            splitter : Splitter instance
                Used to split the dataset for cross-validation folds. By
                convention the first dataset in the tuple returned by the
                splitter is used to train the provided classifier. If the
                first element is 'None' no training is performed. The second
                dataset is used to generate predictions with the (trained)
                classifier.
            combiner : Functor
                Used to aggregate the error values of all cross-validation
                folds.
            harvest_attribs : list of basestr
                What attributes of call to store and return within
                harvested state variable
            copy_attribs : None or basestr
                Force copying values of attributes on harvesting
        """
        DatasetMeasure.__init__(self, **kwargs)
        Harvestable.__init__(self, harvest_attribs, copy_attribs)

        self.__splitter = splitter
        self.__transerror = transerror
        self.__combiner = combiner

# TODO: put back in ASAP
#    def __repr__(self):
#        """String summary over the object
#        """
#        return """CrossValidatedTransferError /
# splitter: %s
# classifier: %s
# errorfx: %s
# combiner: %s""" % (indentDoc(self.__splitter), indentDoc(self.__clf),
#                      indentDoc(self.__errorfx), indentDoc(self.__combiner))


    def _call(self, dataset):
        """Perform cross-validation on a dataset.

        'dataset' is passed to the splitter instance and serves as the source
        dataset to generate split for the single cross-validation folds.
        """
        # store the results of the splitprocessor
        results = []
        self.splits = []

        # what states to enable in terr
        terr_enable = []
        for state_var in ['confusion', 'training_confusion', 'samples_error']:
            if self.states.isEnabled(state_var):
                terr_enable += [state_var]

        # charge states with initial values
        summaryClass = self.__transerror.clf._summaryClass
        self.confusion = summaryClass()
        self.training_confusion = summaryClass()
        self.transerrors = []
        self.samples_error = dict([(id, []) for id in dataset.origids])

        # enable requested states in child TransferError instance (restored
        # again below)
        if len(terr_enable):
            self.__transerror.states._changeTemporarily(
                enable_states=terr_enable)

        # splitter
        for split in self.__splitter(dataset):
            # only train classifier if splitter provides something in first
            # element of tuple -- the is the behavior of TransferError
            if self.states.isEnabled("splits"):
                self.splits.append(split)

            result = self.__transerror(split[1], split[0])

            # next line is important for 'self._harvest' call
            transerror = self.__transerror
            self._harvest(locals())

            # XXX Look below -- may be we should have not auto added .?
            #     then transerrors also could be deprecated
            if self.states.isEnabled("transerrors"):
                self.transerrors.append(copy(self.__transerror))

            # XXX: could be merged with next for loop using a utility class
            # that can add dict elements into a list
            if self.states.isEnabled("samples_error"):
                for k, v in \
                  self.__transerror.states.getvalue("samples_error").iteritems():
                    self.samples_error[k].append(v)

            # pull in child states
            for state_var in ['confusion', 'training_confusion']:
                if self.states.isEnabled(state_var):
                    self.states.getvalue(state_var).__iadd__(
                        self.__transerror.states.getvalue(state_var))

            if __debug__:
                debug("CROSSC", "Split #%d: result %s" \
                      % (len(results), `result`))
            results.append(result)

        # put states of child TransferError back into original config
        if len(terr_enable):
            self.__transerror.states._resetEnabledTemporarily()

        self.results = results
        """Store state variable if it is enabled"""

        return self.__combiner(results)


    splitter = property(fget=lambda self:self.__splitter)
    transerror = property(fget=lambda self:self.__transerror)
    combiner = property(fget=lambda self:self.__combiner)
