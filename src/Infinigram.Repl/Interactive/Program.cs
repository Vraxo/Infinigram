using Infinigram;
using Infinigram.Repl;

TextNormalizationOptions? normalization = null;
return ReplSession.Run(Console.In, Console.Out, Console.Error, "corpus.txt", normalization);
