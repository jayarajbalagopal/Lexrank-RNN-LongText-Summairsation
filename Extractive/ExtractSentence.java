import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ExtractSentence {

    public List<String> count(BreakIterator bi, String source) {
        
    	int counter = 0;
        List<String> sentenceList=new ArrayList<String>();
        /*bi.setText(source);

        int lastIndex = bi.first();
        while (lastIndex != BreakIterator.DONE) {
            int firstIndex = lastIndex;
            lastIndex = bi.next();

            if (lastIndex != BreakIterator.DONE) {
                String sentence = source.substring(firstIndex, lastIndex);
                sentenceList.add(sentence);                
                counter++;
            }
        }
        System.out.println(counter);*/
        

        BreakIterator iterator = BreakIterator.getSentenceInstance(Locale.ENGLISH);
        iterator.setText(source);
        int start = iterator.first();
        int i=1;
        for (int end = iterator.next();end != BreakIterator.DONE; start = end, end = iterator.next()) {
            //System.out.println("Sentence "+i+" : "+source.substring(start,end));
            String sentence = source.substring(start,end);
            sentenceList.add(sentence);
            i++;
        }
        return sentenceList;
    }
    
    public List<String> getSentences(String paragraph){
    	 List<String> sentences;
    	 BreakIterator iterator =BreakIterator.getSentenceInstance(Locale.US);
         
    	 sentences = count(iterator, paragraph);
         
    	
    	
    	 return sentences;
    }
}
