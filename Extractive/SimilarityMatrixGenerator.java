
import java.util.Hashtable;
import java.util.List;

public class SimilarityMatrixGenerator {

List<Hashtable<String,Double>> scoreVector;
double[][] similarityMatrix;
double[] documentWeight;



public SimilarityMatrixGenerator(List<Hashtable<String,Double>> scoreVector)
{
	this.scoreVector=scoreVector;
	similarityMatrix=new double[scoreVector.size()][scoreVector.size()];
    documentWeightCalc();
}


private void documentWeightCalc()
{
	documentWeight=new double[scoreVector.size()];
    for(int j=0;j<documentWeight.length;j++)
		documentWeight[j]=0.0;
    
    for(int i=0;i<documentWeight.length;i++)
	{
				for(double tfidf:scoreVector.get(i).values())
				{
				documentWeight[i]+=Math.pow(tfidf, 2);
				//System.out.print(" "+documentWeight[i]);
				
				}
				//System.out.println();
				
				double temp=Math.sqrt(documentWeight[i]);
				documentWeight[i]=temp;
	}
    for(int i=0;i<scoreVector.size();i++)
    {
            //System.out.print(" "+documentWeight[i]);
    }


	//for(double d:documentWeight)
		//System.out.println(" "+d);
    
   // System.out.println();
	
}

private double cosineXY(int i,int j)
{
	double result=0.0;
	double X,Y,XY;
	X=Y=XY=0.0;
	
	
	for(String word:scoreVector.get(i).keySet())
	{
			if(scoreVector.get(j).containsKey(word))
			XY+=scoreVector.get(i).get(word)*scoreVector.get(j).get(word);
	}
	
	
	
	X=documentWeight[i];
	Y=documentWeight[j];
	
	result=XY/(X*Y);
	
	
	return result;
}


public double[][] similarityMatrixGenerator()
{
	int N=scoreVector.size();
	similarityMatrix=new double[N][N];
	int i=0;
	for(i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
			similarityMatrix[i][j]=0.0;
	}
	
	for(i=0;i<N;i++)
		similarityMatrix[i][i]=1;
	
	
	for(i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			
				if(i!=j)
				similarityMatrix[i][j]=cosineXY(i, j);
		
		}
	}
   /* for(i=0;i<scoreVector.size();i++)
    {
        for(int j=0;j<scoreVector.size();j++)
            //System.out.print(" "+similarityMatrix[i][j]);
        System.out.println();
    }*/
    //System.out.println();
    //System.out.println();

	return similarityMatrix;
}


}
