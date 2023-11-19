# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease

Python code for the paper:

Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by 
Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023

This code is "broken research code", and thus the following disclaimer appllies: 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Code usage
1) setup.py is the main configuration file. It should be modified first. This file uses dataLoader.py to perform the actual loading of the files.
2) Prepro.py should be executed next, as it computes the preprocessing of the model (computation of G, we in the code) and uses the previous one.
3) plotPrepro.py plots the results of Prepro.py. Should be done AFTER Prepro.py.
4) Finally, pipeline2.py should be executed. This file was written before the big re-factorization of the WholeBrain library, so it partially uses its newer features. Also, it uses the G=3.1 value found at Prepro.py.
5) plot3DBrainBurdens is used to plot the nice 3D plots ocmparing ABeta and Tau for HC, MCI and AD.
6) Shuffling.py is used for the comparisons with shuffled random data...
7) plotParmComparisonAcrossCohort.py is used for plotting the different figures in the paper, see comments within at the respective boolean variables...

If the AT(N) classification should be used, first run classific_AT(N).py, and then configure and use setup_ATN.py. All the rest of the system should continue working as before (this only affects the final cohort grouping for plotting).