import java.util.Hashtable;
import java.util.List;
import java.util.Collections;
import java.io.*;
public class Controller {
	
	ExtractSentence extractSentence;
	Lexrank lexrank;
	List<Hashtable<String,Double>> score;
	int [] degree;
	public static String filename;
		
	public String getSummary() throws Exception{
		
		String summary = " ";
		String preprocessedText=null;
		List<String> sentences=null;
		
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		StringBuilder stringBuilder = new StringBuilder();
		String line = null;
		String ls = System.getProperty("line.separator");
		while ((line = reader.readLine()) != null) {
			stringBuilder.append(line);
			stringBuilder.append(ls);
		}

		// delete the last new line separator
		stringBuilder.deleteCharAt(stringBuilder.length() - 1);
		reader.close();

		String content = stringBuilder.toString();

		String summarizableText = content;

		if(!summarizableText.equals("")){
			
			preprocessedText = summarizableText;
			
		}
		else{
			throw new Exception(Constants.TEXT_EXTRACTION_FAILED);
		}
		
		extractSentence = new ExtractSentence();
		sentences = extractSentence.getSentences(preprocessedText);
        int count;
       
        count=sentences.size();
       

        
       double[][] similarityMatrix = getSimilarityMatrixFromSentenceList(sentences);
     
        
		lexrank = new Lexrank(sentences, similarityMatrix	,0.85, degree);
		double[] resultEigenValues = lexrank.powerMethod(.001);
		
        int sample[]=new int[sentences.size()];
        
        for(int i=0;i<sentences.size();i++)
        	{
        		sample[i]=i;
        	}
        for (int i = 0; i < sentences.size()-1; i++)
            for (int j = 0; j < sentences.size()-i-1; j++)
                if (resultEigenValues[j] < resultEigenValues[j+1])
                {
                    double temp = resultEigenValues[j];
                    resultEigenValues[j] = resultEigenValues[j+1];
                    resultEigenValues[j+1] = temp;
                    Collections.swap(sentences,j,j+1);
                }
        
       
        int k=1;
        double x=(0.4*count);
        int max=(int)x;
        int m=1;
        for(String s:sentences)
        	{
                if(k<=max)
                {
                    summary=summary+s;
                    m++;
                    k++;
                }
                else
                    ;
            }
		return summary;
		
	}
	
	
	private void printArray(double[] A){
		
		for(double i: A){
			System.out.print(" "+i);
		}
		
	}
	private void printArray(int[] A){
		
		for(int i: A){
			System.out.print(" "+i);
		}
		
	}
	
	private double[][] getSimilarityMatrixFromSentenceList(List<String> sentenceList){
		
		
		DocumentVectorCalculator dvc=new DocumentVectorCalculator(sentenceList);
        
		score = dvc.generateTF_IDFScores();
        
		SimilarityMatrixGenerator simg=new SimilarityMatrixGenerator(score);
		
		double[][] result=simg.similarityMatrixGenerator();
        //result contains the similarity matrix ..in this case a [30][30] matrix
		degree = new int[result.length];
		
		for(int i=0;i<degree.length;i++)
        {
			degree[i] = 0;
		}
        
        
		
			for(int i=0;i<result.length;i++)
			{
				for(int j=0;j<result[i].length;j++)
				{
					
					if(i!=j)
					{
						if(result[i][j] >= 0.1)
						{result[i][j]=1.0;degree[i]++;}
						else
						result[i][j]=0.0;
					}
				}
			}
		

		
		//End
			for(int i=0;i<result.length;i++)
			{
				for(int j=0;j<result[i].length;j++)
				{
                    if(degree[i]!=0)
                    {
                        result[i][j]=result[i][j]/degree[i];

                    }
                    else
                    {
                        result[i][j]=0;

                    }
				}
			}
			
		return result;
		
	}
	public static void main(String...a) throws Exception{

		filename = a[0];
		Controller c = new Controller();

		PrintWriter writer = new PrintWriter("../Outputs/out.txt", "UTF-8");
		writer.println(c.getSummary());
		writer.close();
	}


}
