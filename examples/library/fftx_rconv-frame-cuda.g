
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

# 1d and multidimensional complex DFTs

##  Script to generate code, will be driven by a size specification and will write the
##  CUDA code to a file.  The code will be compiled along with a test harness to run the
##  code, timing it against a cufft specification of the same size, and validating that
##  the results are the same for both.

Load(fftx);
ImportAll(fftx);

# startup script should set LocalConfig.fftx.defaultConf() -> LocalConfig.fftx.confGPU() 
# conf := LocalConfig.fftx.defaultConf();  
conf := LocalConfig.fftx.confGPU();

if 1 = 1 then
    d := Length(szcube);
    szstring := StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    ## name := "fftx_rconv"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    ## Drop the dimensionality from source file name.
    name := "fftx_rconv"::"_"::szstring;
    
    PrintLine("fftx_rconv-cuda-frame: name = ", name, ", cube = ", szcube, ", size = ",
              StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1),
                                                                    s->" x "::StringInt(s))),
              ";\t\t##PICKME##");

    ## This line from mddft-frame-cuda.g :
    ##    t := TFCall(TRC(MDDFT(szcube, 1)), 
    ##                rec(fname := name, params := []));
    szrevcube := Reversed(szcube);
    szhalfcube := [szcube[1]/2+1]::Drop(szcube,1);
    var_1:= var("var_1", BoxND(szhalfcube, TReal));
    var_2:= var("var_2", BoxND(szhalfcube, TReal));
    var_3:= var("var_3", BoxND(szcube, TReal));
    var_4:= var("var_4", BoxND(szcube, TReal));
    var_5:= var("var_5", BoxND(szhalfcube, TReal));
    var_3:= X;
    var_4:= Y;
    symvar := var("sym", TPtr(TReal));
    t := TFCall(TDecl(TDAG([
        TDAGNode(MDPRDFT(szrevcube,-1), var_1,var_3),
        TDAGNode(Diag(diagTensor(FDataOfs(symvar,Product(szhalfcube),0),fConst(TReal, 2, 1))), var_2,var_1),
        TDAGNode(IMDPRDFT(szrevcube,1), var_4,var_2),
                  ]),
            [var_1,var_2]
            ),
        rec(fname:="fftx_rconv_"::szstring, params:= [symvar])
    );
    prefix:="fftx_rconv";

    
    opts := conf.getOpts(t);
    ##  PrintLine("DEBUG: opts = ", opts);
    ##  opts.printRuleTree := true;
    if IsBound ( seedme ) then 
        RandomSeed ( seedme );
    fi;
    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;

    ##  We need to functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;
    tt := opts.tagIt(t);
    if(IsBound(fftx_includes)) then opts.includes:=fftx_includes; fi;
    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo(libdir::"/"::name::file_suffix, opts.prettyPrint(c));
fi;
