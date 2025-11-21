\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{antgroup}
\LoadClassWithOptions{article}

\RequirePackage[top=2.75cm, bottom=2.5cm, left=2.5cm, right=2.5cm, columnsep=0.65cm]{geometry}
\RequirePackage{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[latin, english]{babel}
\RequirePackage{lipsum}
\RequirePackage{etoolbox}
\RequirePackage{enumitem} 
\RequirePackage{fancyhdr}
%\DisableLigatures[f]{family=sf*} 
\RequirePackage{graphicx}
\RequirePackage{subcaption}
\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}
\RequirePackage[most]{tcolorbox}
\RequirePackage{xcolor}
\definecolor{seedbg}{HTML}{2E5AA8}
\definecolor{seedblue}{HTML}{2E5AA8}
\newcommand{\seedblue}[1]{{\bfseries\color[HTML]{2E5AA8} #1}} 
\definecolor{black}{rgb}{0,0,0} 
\newcommand{\black}[1]{{\bfseries\color[HTML]{000000} #1}}

\addto{\captionsenglish}{\renewcommand{\refname}{References}}
\newcommand{\citeas}[1]{\Needspace*{8\baselineskip}\noindent\bibentry{#1}}



\newcommand{\sectionfont}
{\fontsize{12}{10}\selectfont} 
\newcommand{\subsectionfont}
{\fontsize{11}{10}\selectfont} 
\RequirePackage{titlesec}
\titleformat*{\paragraph}{\itshape}


\newcommand{\bffont}
{\fontsize{8.8}{10}\selectfont} 

\renewcommand\addtolist[5][]{
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect 
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}

\newcommand{\authorfont}
{\fontsize{10}{12}\selectfont} 

\newcommand\authorlist{}
\newcommand\authorformat[2][]{\authorfont {\textbf{#2}$^{#1}$}}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{}}


\newcommand{\affiliationfont}
{\fontsize{10}{12}}

\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\affiliationfont {$^{#1}$#2}}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{}}



\newcommand{\contributionfont}
{\fontsize{10}{12}\selectfont}

\newcommand\contributionlist{}
\newcommand\contributionformat[2][]{{\contributionfont $^{#1}$#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{}}

\newcommand{\checkdatafont}
{\fontsize{7.92}{10}\selectfont}

\newcommand\checkdatalist{}
\newcommand\checkdataformat[2][]{{\small {\bfseries #1:} #2}}
\newcommand\checkdata[2][]{\addtolist[#1]{#2}{\checkdatalist}{\checkdataformat}{\par}}

