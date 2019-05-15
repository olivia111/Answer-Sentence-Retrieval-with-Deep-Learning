using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace squad_concat
{
    class Program
    {
        
        static void Main(string[] args)
        {
            //dedup("process.train.data", "process.train.dedup.data");
            //concat("process.train.dedup.data", "squad.process.train.dedup.concat.data");
            //convertAnswerSentence("squad.process.train.dedup.concat.data.tsv", "squad.process.train.dedup.concat.convert.data.tsv");
            convertAnswerSentenceMarco("marco-proces-train-data.tsv", "marco-process-train-data-convert.tsv");
        }

        public static void convertAnswerSentenceMarco(string inputFile, string outputFile)
        {
            StreamReader sr = new StreamReader(inputFile);
            StreamWriter sw = new StreamWriter(outputFile);

            int lineCnt = 0;
            while (!sr.EndOfStream)
            {
                var line = sr.ReadLine();
                var seg = line.Split('\t');
                if (seg.Length != 5)
                {
                    continue;
                }

                var query = seg[0];
                var url= seg[1];
                var passage = seg[2];
                var answerIndex = seg[3];
                var doc = seg[4];

                var answerSentence = string.Empty;

                if (string.IsNullOrEmpty(answerIndex))
                {
                    continue;
                }

                var answerIndexSeg = answerIndex.Split(new char[] { ',', ' ' },StringSplitOptions.RemoveEmptyEntries);
                var answerText = new List<string>();
                int count = answerIndexSeg.Length / 2;
                for(var i=0; i< count; i++)
                {
                    int startIndex = Convert.ToInt32( answerIndexSeg[i]);
                    int endIndex = Convert.ToInt32( answerIndexSeg[i + 1]);
                    if (startIndex > endIndex) continue;
                    if (endIndex >= passage.Length) endIndex = passage.Length - 1;
                    var ans = passage.Substring(startIndex, (endIndex - startIndex + 1));
                    if (string.IsNullOrEmpty(ans)) continue;
                    answerText.Add(ans);
                }


                var passageSeg = passage.Split(new char[] { '.', '?', '!' });
                for (var i = 0; i < passageSeg.Length; i++)
                {
                    foreach (var ans in answerText)
                    {
                        var ansText = ans.Replace(".", "").Replace("?", "").Replace("!", "");
                        if (passageSeg[i].Contains(ansText) && doc.Contains(ans))
                        {
                            answerSentence += passageSeg[i]+".";
                            //break;
                        }
                    }
                }

                if (!string.IsNullOrEmpty(answerSentence))
                {
                    sw.WriteLine(query + "\t" + answerSentence + "\t" + doc);
                    lineCnt++;
                }
            }
            

            sr.Close();
            sw.Close();

            //Console.Read();
        }

        public static void convertAnswerSentence(string inputFile, string outputFile)
        {
            StreamReader sr = new StreamReader(inputFile);
            StreamWriter sw = new StreamWriter(outputFile);

            while(!sr.EndOfStream)
            {
                var line = sr.ReadLine();
                var seg = line.Split('\t');
                if(seg.Length != 5)
                {
                    continue;
                }

                var query = seg[0];
                var answerStartIndex = seg[1];
                var answerText = seg[2];
                var passage = seg[3];
                var doc = seg[4];

                var answerSentence = string.Empty;

                if(string.IsNullOrEmpty(answerStartIndex) || answerStartIndex =="-a")
                {
                    continue;
                }

                int answerStartId = Convert.ToInt32(answerStartIndex);

                var passageSeg = passage.Split(new char[] {'.', '?', '!' });
                for(var i = 0; i<passageSeg.Length; i++)
                {
                    if(passageSeg[i].Contains(answerText))
                    {
                        answerSentence = passageSeg[i]+".";
                        break;
                    }
                }

                if(!string.IsNullOrEmpty(answerSentence))
                {
                    sw.WriteLine(query + "\t" + answerSentence + "\t" + doc);
                }
            }

            sr.Close();
            sw.Close();
        }

        public static void concat(string inputFile, string outputFile)
        {
            StreamReader sr = new StreamReader(inputFile);
            StreamWriter sw = new StreamWriter(outputFile);

            var lineDict = new List<string>();
            var docDict = new Dictionary<string, string>();
            var docStr = string.Empty;
            var docStrSet = new HashSet<string>();
            var preDocId = string.Empty;
            while(!sr.EndOfStream)
            {
                var line = sr.ReadLine();
                if(string.IsNullOrEmpty(line))
                {
                    continue;
                }

                lineDict.Add(line);
                var seg = line.Split('\t');
                if(seg.Length!=5)
                {
                    continue;
                }
                
                if(preDocId==string.Empty || seg[0]!=preDocId)
                {
                    if(!string.IsNullOrEmpty(docStr))
                    {
                        docDict.Add(preDocId, docStr);
                    }

                    docStrSet = new HashSet<string>();
                    preDocId = seg[0];
                    docStr = seg[1];
                    docStrSet.Add(seg[1]);
                }
                else
                {
                    if(preDocId == seg[0])
                    {
                        if (docStrSet.Contains(seg[1])) continue;

                        docStr += " " + seg[1];
                        docStrSet.Add(seg[1]);
                    }
                }
            }

            if (!string.IsNullOrEmpty(docStr))
            {
                docDict.Add(preDocId, docStr);
            }


            for (var i=0; i< lineDict.Count; i++)
            {
                var seg = lineDict[i].Split('\t');
                if (seg.Length != 5)
                {
                    continue;
                }

                if (docDict.ContainsKey(seg[0]))
                {
                    sw.WriteLine(seg[2]+"\t"+seg[3]+"\t" + seg[4] +"\t" + seg[1] + "\t" + docDict[seg[0]]);
                }
            }

            sr.Close();
            sw.Close();
        }

        public static void dedup(string inputFile, string outputFile)
        {
            StreamReader sr = new StreamReader(inputFile);
            StreamWriter sw = new StreamWriter(outputFile);

            HashSet<string> dict = new HashSet<string>();

            while(!sr.EndOfStream)
            {
                var line = sr.ReadLine();
                if(string.IsNullOrEmpty(line))
                {
                    continue;
                }

                if(dict.Contains(line))
                {
                    continue;
                }

                sw.WriteLine(line);
                dict.Add(line);
            }

            sr.Close();
            sw.Close();
        }
    }
}
