## How to ins libs

mkdir -p \
data/raw data/processed data/external \
notebooks \
src/data src/features src/models src/utils \
app \
configs \
scripts \
tests \
.github/workflows \
docker \
great_expectations \
mlruns \
artifacts

--------------------------------------------------------

python -m venv .deployment
source .deployment/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt


