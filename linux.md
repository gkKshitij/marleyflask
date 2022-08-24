git clone https://github.com/gkKshitij/marleyflask.git
cd marleyflask
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
chmod -v +x Miniconda*.sh
echo "4dc4214839c60b2f5eb3efbdee1ef5d9b45e74f2c09fcae6c8934a13f36ffc3e *Miniconda3-py37_4.12.0-Linux-x86_64.sh" | shasum --check
./Miniconda3-py39_4.12.0-Linux-x86_64.sh

#add below lines to ~./bashrc
if ! [[ $PATH =~ "$HOME/miniconda3/bin" ]]; then
  PATH="$HOME/miniconda3/bin:$PATH"
fi

conda list
done



conda create --name pyis -c anaconda python=3.6 anaconda=5.2.0


apt-get update && apt-get install libgl1
conda install -c conda-forge opencv
conda install -c conda-forge statistics
conda install -c conda-forge flask
conda install -c conda-forge imutils
conda install -c conda-forge scipy
conda install -c conda-forge matplotlib-base

