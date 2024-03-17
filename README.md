# Estimating Molecule Solubility in water using GNN in Pytorch
Using Graph Neural Networks for estimating water solubility of a molecule structure using Pytorch.

Dataset:  ESOL is a water solubility prediction dataset consisting of 1128 samples.

![structure image](https://github.com/mr-ravin/Molecule-Solubility-using-GNN-in-Pytorch/blob/main/structure.png?raw=true)

#### File Structure
```
|-- dataset/
|-- weights/
|-- result/
|-- main.py
|-- gnn_model.py
|-- utils.py
|-- requirements.txt
```

### Training
```python
python3 main.py --mode train --epoch 2000
```
![train image](https://github.com/mr-ravin/Molecule-Solubility-using-GNN-in-Pytorch/blob/main/result/training_analysis.png?raw=true)

### Testing
```python
python3 main.py --mode test
```
![test image](https://github.com/mr-ravin/Molecule-Solubility-using-GNN-in-Pytorch/blob/main/result/testing_analysis.png?raw=true)

```
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
