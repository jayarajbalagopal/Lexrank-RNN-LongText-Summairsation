
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.StringTokenizer;


public class DocumentVectorCalculator {

List<String> documentList;
public List<Hashtable<String,Double>> scoreVector;


public DocumentVectorCalculator(List<String> documentList)
{
	this.documentList=documentList;
	scoreVector=new ArrayList<Hashtable<String,Double>>(this.documentList.size());
}


public void generateTFScores()
{
	for(String t:documentList)
	{
	
		StringTokenizer stNew=new StringTokenizer(t," ");
		Hashtable<String, Double> tempHash=new Hashtable<String,Double>();
		double max=1.0;
		
		while(stNew.hasMoreTokens())
		{
				String token=stNew.nextToken();
            
		
				
				if(tempHash.containsKey(token))
                    tempHash.put(token, Double.valueOf((tempHash.get(token)+1)));
				else
                    tempHash.put(token, 1.0);
			
                double freq=tempHash.get(token);
                if(max<freq)
                    max=freq;
		
		
		}
		//System.out.println(tempHash.toString());
		//System.out.println("Max="+max);
		
		
		for(String key:tempHash.keySet())
		{
			tempHash.put(key, tempHash.get(key)/max);
		}
			
		
		
		scoreVector.add(tempHash);
        //System.out.println(scoreVector);
    }
	
}

private double calcIDF(String word)
{
	int rho=documentList.size();
	int rhow=0;
	
	
	for(Hashtable<String,Double> temp:scoreVector)
	{
		if(temp.containsKey(word))
			rhow++;
	}
	
	//System.out.println(word+" rho="+rho+" rhow="+rhow);
	
	return Math.log((double)rho/(double)rhow);
}

/*
 * Generates TF-IDF scores
 * 
 */
public List<Hashtable<String,Double>> generateTF_IDFScores()
{
	generateTFScores();

	//System.out.println("After only TF Scores");
	//System.out.println(scoreVector.toString());
	
	for(Hashtable<String,Double> tempHash:scoreVector)
	{
		
		for(String key:tempHash.keySet())
		{
			tempHash.put(key, tempHash.get(key)*calcIDF(key));
			//System.out.println(key+" =>"+calcIDF(key));
			
		}
		
			
	}
	

	return scoreVector;
}

}
