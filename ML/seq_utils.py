import os
import pandas as pd

from Bio import SeqIO,AlignIO

import textwrap
import tempfile
import subprocess

import numpy as np
from scipy.spatial.distance import squareform, pdist

from IPython.display import display


output_path = os.getcwd()

alphabet = "ARNDCQEGHILKMFPSTWYV-"
states = len(alphabet)
a2n = {}
for a, n in zip(alphabet, range(states)):
    a2n[a] = n
################




def get_output_dir(out_path=output_path):

    return out_path


def aa2num(aa):
    """convert aa into num"""
    if aa in a2n:
        return a2n[aa]
    else:
        return a2n["-"]


def view_seqs(fasta) -> None:
    """Display fasta sequences in-cell & in-memory.

    :param df: Input pandas dataframe.
    :type df: pd.DataFrame
    :param id_col: Name of column with molecule ID information.
    :type id_col: str
    :param start: Index of first amino acid position column.
    :type start: int
    :param end: Index of last amino acid position column.
    :type end: int
    """

    bundle = {}
    bundle["application/vnd.fasta.fasta"] = fasta
    bundle["text/plain"] = fasta
    display(bundle, raw=True)


def read_fasta_file(fname: str, frmt: str = "fasta", to_fasta_fmt=True, view=False):
    """[summary]

    Parameters
    ----------
    fname : str
        [description]
    frmt : str, optional
        [description], by default "fasta"
    """ """
    Parameters
    ----------
    filename : str
        [description]
    """

    records = SeqIO.parse(fname, frmt)
    

    if to_fasta_fmt:
        seq_list = []
        header_list = []

        for record in records:
            seq_list.append(str(record.seq))
            header_list.append(str(record.id))
        
        print('Read',len(seq_list),'sequences')

        fasta_fmt = list_2_fasta_fmt2(seq_list, header_list, view=view)
        
        records=fasta_fmt

    return records


def show_fasta_file(fname, frmt="fasta"):

    fasta = read_fasta_file(fname, frmt,view=True)

    return


def list_2_fasta_fmt(
    seq_list: list, header_list: list, view: bool = False, line_wrapper_length: int = 60
) -> str:

    """Generate a fasta format text from a list of sequence strings

    INPUT:
        seq_list :  list of sequence strings

        Optional:
            header_list  : id for the sequence for file_out (Default : None, based on counter)
            view         : display sequences with MSAviewer (Default : False)

    OUTPUT:
        fasta format text string

    """

    fasta = []

    if header_list is None:
        header_list = ["seq_" + str(i) for i in range(0, len(seq_list))]

    for header, sequence in zip(header_list, seq_list):

        fasta.append(">{}\n".format(header))

        for line in textwrap.wrap(sequence, line_wrapper_length):
            fasta.append("{}\n".format(line))

    if view:
        view_seqs(fasta)

    return fasta

def list_2_fasta_fmt2(
    seq_list: list, header_list: list, view: bool = False, line_wrapper_length: int = 60
) -> str:

    """Generate a fasta format text from a list of sequence strings

    INPUT:
        seq_list :  list of sequence strings

        Optional:
            header_list  : id for the sequence for file_out (Default : None, based on counter)
            view         : display sequences with MSAviewer (Default : False)

    OUTPUT:
        fasta format text string

    """

    fasta = ''

    if header_list is None:
        header_list = ["seq_" + str(i) for i in range(0, len(seq_list))]

    for header, sequence in zip(header_list, seq_list):

        fasta=fasta+(">{}\n".format(header))

        for line in textwrap.wrap(sequence, line_wrapper_length):
            fasta=fasta+("{}\n".format(line))

    if view:
        view_seqs(fasta)

    return fasta


def fasta_2_file(fasta, file_out="tmp", out_dir="", verbose=True):
    """Generate a fasta file from a list of sequence strings

    INPUT:
        seq_list :  list of sequence strings

        Optional:
            header_list  : id for the sequence for file_out (Default : None, based on counter)
            file_out     : output filename                  (Default : 'tmp', fname is auto-created )
            out_dir      : output folder                    (Default : '' )
            verbose      : display filename (Default : True)

    OUTPUT:
        fasta filename with full path

    """

    if file_out == "tmp" or file_out is None:
        file_tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        file_out = file_tmp.name
        out_dir = ""

    f_output = os.path.join(out_dir, file_out)

    if verbose:
        print("Fasta file saved in", f_output)

    with open(f_output, "w") as fasta_file:
        #print('LEN ',len(fasta),type(fasta),fasta)
        for line in fasta:
            #print(type(line),line)
            fasta_file.write(str(line))

    return f_output


def seqs_list_2_fasta(
    seq_list,
    header_list=None,
    file_out=None,
    return_fname=False,
    out_dir=output_path,
    view=False,
    verbose=True,
):
    """Generate a fasta file from a list of sequence strings

    INPUT:
        seq_list :  list of sequence strings

        Optional:
            header_list  : id for the sequence for file_out (Default : None, based on counter)
            file_out     : output filename                  (Default : None )
            out_dir      : output folder                    (Default : output_path )
            view         : display sequences with MSAviewer (Default : False)
            verbose      : display filename if called       (Default : True)

    OUTPUT:
        output file_name

    """

    fasta_fmt = list_2_fasta_fmt2(seq_list, header_list, view=view)

    if file_out is not None:

        if (not os.path.exists(out_dir)) & (file_out != "tmp"):
            os.makedirs(out_dir)

        f_output = fasta_2_file(
            fasta_fmt, file_out=file_out, out_dir=out_dir, verbose=verbose
        )

        if return_fname:
            fasta_fmt = f_output

    return fasta_fmt

    

def read_aln_file(fname, frmt="fasta"):

    align = AlignIO.read(fname, frmt)
    
    print(len(align),'sequences of length',len(str(align[0].seq)))
    
    return align


def show_alignment(fname, frmt="fasta"):

    align = read_aln_file(fname, frmt)
    aligned_fasta = align_2_fasta_fmt(align, view=True)

    return


def align_2_fasta_fmt(align, view: bool = False, line_wrapper_length: int = 60):
    """[summary]

    Parameters
    ----------
    align : [type]
        [description]
    view : bool, optional
        [description], by default False
    line_wrapper_length : int, optional
        [description], by default 60

    Returns
    -------
    [type]
        [description]
    """
    fasta = ''

    for record in align:

        fasta=fasta+(">{}\n".format(str(record.id)))

        for line in textwrap.wrap(str(record.seq), line_wrapper_length):
            fasta=fasta+("{}\n".format(line))

    if view:
        view_seqs(fasta)

    return fasta


def align_file(
    file_in,
    file_out=None,
    options="",
    anysymbol=True,
    remove=False,
    view=False,
    return_type="seq_list",
):

    if anysymbol:
        options += "--anysymbol "

    if file_out == "tmp" or file_out is None:
     #   file_tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False)
      #  file_out = file_tmp.name
        out_dir = ""
        remove = True

    # f_output = os.path.join(out_dir, file_out)

    cmd = "mafft --quiet " + options + file_in
    # print(cmd)
    mafft_aln = subprocess.run(cmd.split(), check=True, capture_output=True,text=True)
    #os.system(cmd)
    #print(mafft_aln.stdout)
    
    file_out=fasta_2_file(mafft_aln.stdout,file_out=file_out)
    


    align = read_aln_file(file_out)
    
    #print(len(align),'sequence of length',len(str(align[0].seq)))

    if return_type == "fasta_fmt" or view:

        aligned_fasta = align_2_fasta_fmt(align, view=view)

    if remove and return_type != "fname":
        # os.remove(file_in)
        os.remove(file_out)

    if return_type == "fasta_fmt":
        aligned_seqs = aligned_fasta

    elif return_type == "fname":
        aligned_seqs = file_out
    
    elif return_type == "AlignIO":
        aligned_seqs = align

    else:
        aligned_seqs = [str(s.seq) for s in align]  # , [str(s.id) for s in align]

    return aligned_seqs


def align_seq_list(
    seq_list,
    header_list=None,
    file_out=None,
    remove=False,
    aln_options="",
    view=False,
    return_type="seq_list",
):

    """Align AA sequences from a list.

    Arguments:
        seq_list -- list of strings

        Optional:
            header_list  : id for the sequence for file_out (Default : None )
            file_out     : output filename                  (Default : None )

    Returns:
        list of aligned sequences (as strings with same lenghts)
        Optional a fasta file (file_out)

    """

    file_in = seqs_list_2_fasta(
        seq_list, header_list, file_out="tmp", return_fname=True
    )

    aligned_seqs = align_file(
        file_in, file_out, options=aln_options, view=view, return_type=return_type
    )

    os.remove(file_in)

    # new_seq_list = [str(s.seq) for s in aligned_seqs]  # , [str(s.id) for s in align]

    return aligned_seqs


def align_df_seq(
    df,
    seq_label="Sequence",
    header_label=None,
    file_out=None,
    aln_options="",
    view=False,
    return_type="seq_list",  # 'fname' # 'fasta_fmt'
):

    """Align AA sequences from a DF.

    Arguments:
        df -- DataFrame

        Optional:
            seq_label    : df fields for sequence retrieval (Default : 'Sequence' )
            header_list  : id for the sequence for file_out (Default : None )
            file_out     : output filename                  (Default : None )



    Returns:
        list of sequences (as strings with same lenghts)

    """

    if header_label is None:
        header_list = df.index.tolist()
    else:
        header_list = df[header_label].tolist()

    output = align_seq_list(
        df[seq_label],
        header_list,
        file_out,
        aln_options=aln_options,
        view=view,
        return_type=return_type,
    )

    return output

def mafft_distances(seqs_list, pair_distance="--localpair", remove=True, square=False):
    # sequence distances without alignment
    # pair_distance can be "--localpair", "--globalpair" or "" for k-mers (similar to hamming)

    tmp_fasta = seqs_list_2_fasta(
        seqs_list, file_out="tmp", return_fname=True
    )

    cmd = (
        "mafft --quiet --retree 0 --distout "
        #"/Users/laila/opt/anaconda3/bin/mafft --quiet --retree 0 --distout "
        + pair_distance
        + " "
        + tmp_fasta
    )

    print(
        "\nComputing pairwise distances "
        + pair_distance
        + " for "
        + str(len(seqs_list))
        + " sequences"
    )
    a = subprocess.run(cmd.split(), check=True, capture_output=True)

    # os.system(cmd)

    cmd2 = "awk '{if(NR==2)n=$1;if(NR>n+3) printf $0}' " + (tmp_fasta) + ".hat2 "

    distances = np.array(os.popen(cmd2).read().split(), dtype="f8")
    # distances = np.loadtxt(os.getcwd()+"/dist_out.txt", comments="#", unpack=False)

    os.remove(tmp_fasta)

    if remove:
        os.remove(tmp_fasta + ".hat2")

    if square:
        distances = squareform(distances)

    return distances


def hamming_distances(seqs, normalize=False, square=False):

    data = []

    for seq in seqs:
        data.append([(ord(x)) for x in seq])
    
    distances = pdist(data, "hamming")

    if not normalize:
        distances = distances * len(seq)

    if square:
        distances = squareform(distances)

    return distances


amino_acids_1 = np.array(['A','R','N','D','C','E','Q','G','H','I',\
'L','K','M','F','P','S','T','W','Y','V'])


''' Z5-scales according to Sandberg et al., J Med Chem, 1998 '''
z_scales_1 = np.array([[ 0.24, 3.52, 3.05, 3.98, 0.84, 1.75, 3.11, 2.05, 2.47,-3.89,\
-4.28, 2.29,-2.85,-4.22,-1.66, 2.39, 0.75,-4.36,-2.54,-2.59],
[-2.32, 2.5 , 1.62, 0.93,-1.67, 0.5 , 0.26,-4.06, 1.95,-1.73,\
-1.30, 0.89,-0.22, 1.94, 0.27,-1.07,-2.18, 3.94, 2.44,-2.64],
[ 0.6 ,-3.50, 1.04, 1.93, 3.71,-1.44,-0.11, 0.36, 0.26,-1.71,\
-1.49,-2.49, 0.47, 1.06, 1.84, 1.15,-1.12, 0.59, 0.43,-1.54],
[-0.14 ,1.99,-1.15,-2.46, 0.18,-1.34,-3.04,-0.82, 3.9 ,-0.84,\
-0.72, 1.49, 1.94, 0.54, 0.7 ,-1.39,-1.46, 3.44, 0.04,-0.85],
[ 1.3 ,-0.17, 1.61, 0.75,-2.65, 0.66,-0.25,-0.38, 0.09, 0.26, \
0.84, 0.31,-0.98,-0.62, 2 , 0.67,-0.40,-1.59,-1.47,-0.02]])



# Function to encode amino acids as z-scales
def get_z_scales(code, z_scales=z_scales_1, z=3):

    """
    Parameters
    ----------
    code : str # residue code
    amino_acid_codes : np.array # list of residue codes (i.e. 1 letter code)
    z : int # number of output Z-scales per amino acid



    Returns
    -------
    np.array # fingerprint of length z (i.e. z = {3, 5})
    """

    amino_acid_codes=amino_acids_1

    if z!=3 and z!=5:
        print('z=%s! z must be 3 or 5' % str(z))

    z_scales_val=[]

    if code.upper() in amino_acid_codes:
        # get first 3 or 5 value for a particular amino acid
        z_scales_val = z_scales.T[np.where(code.upper() == amino_acid_codes)[0]][0][:z]

    elif code == "0" or code == '-' or code == 'X' or code == '*':
        z_scales_val = [0]*z

    else:
        print('%s Not an amino acid' % str(code))

    return (z_scales_val)


def transla_peptide(list_seqs, z_scales=z_scales_1, z=3):
    trans=[]

    for seq in list_seqs:
        trans.append((np.concatenate( [get_z_scales(q, z_scales, z=z) for q in seq])))

    return(trans)
