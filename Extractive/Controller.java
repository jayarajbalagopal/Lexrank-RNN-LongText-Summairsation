import java.util.Hashtable;
import java.util.List;
import java.util.Collections;
public class Controller {
	
	ExtractSentence extractSentence;
	Lexrank lexrank;
	List<Hashtable<String,Double>> score;
	int [] degree;
		
	public String getSummary() throws Exception{
		
		String summary = " ";
		String preprocessedText=null;
		List<String> sentences=null;
		
        //String summarizableText ="Parallel computing is a type of computation in which many calculations or the execution of processes are carried out simultaneously. Large problems can often be divided into smaller ones, which can then be solved at the same time. There are several different forms of parallel computing: bit-level, instruction-level, data, and task parallelism. Parallelism has been employed for many years, mainly in high-performance computing, but interest in it has grown lately due to the physical constraints preventing frequency scaling. As power consumption (and consequently heat generation) by computers has become a concern in recent years,parallel computing has become the dominant paradigm in computer architecture, mainly in the form of multi-core processors. Parallel computing is closely related to concurrent computing—they are frequently used together, and often conflated, though the two are distinct: it is possible to have parallelism without concurrency (such as bit-level parallelism), and concurrency without parallelism (such as multitasking by time-sharing on a single-core CPU). In parallel computing, a computational task is typically broken down in several, often many, very similar subtasks that can be processed independently and whose results are combined afterwards, upon completion. In contrast, in concurrent computing, the various processes often do not address related tasks; when they do, as is typical in distributed computing, the separate tasks may have a varied nature and often require some inter-process communication during execution. Parallel computers can be roughly classified according to the level at which the hardware supports parallelism, with multi-core and multi-processor computers having multiple processing elements within a single machine, while clusters, MPPs, and grids use multiple computers to work on the same task. Specialized parallel computer architectures are sometimes used alongside traditional processors, for accelerating specific tasks. In some cases parallelism is transparent to the programmer, such as in bit-level or instruction-level parallelism, but explicitly parallel algorithms, particularly those that use concurrency, are more difficult to write than sequential ones,because concurrency introduces several new classes of potential software bugs, of which race conditions are the most common. Communication and synchronization between the different subtasks are typically some of the greatest obstacles to getting good parallel program performance. A theoretical upper bound on the speed-up of a single program as a result of parallelization is given by Amdahl's law. Traditionally, computer software has been written for serial computation. To solve a problem, an algorithm is constructed and implemented as a serial stream of instructions. These instructions are executed on a central processing unit on one computer. Only one instruction may execute at a time—after that instruction is finished, the next one is executed. Parallel computing, on the other hand, uses multiple processing elements simultaneously to solve a problem. This is accomplished by breaking the problem into independent parts so that each processing element can execute its part of the algorithm simultaneously with the others. The processing elements can be diverse and include resources such as a single computer with multiple processors, several networked computers, specialized hardware, or any combination of the above. Frequency scaling was the dominant reason for improvements in computer performance from the mid-1980s until 2004. The runtime of a program is equal to the number of instructions multiplied by the average time per instruction. Maintaining everything else constant, increasing the clock frequency decreases the average time it takes to execute an instruction. An increase in frequency thus decreases runtime for all compute-bound programs. However, power consumption P by a chip is given by the equation P = C × V 2 × F, where C is the capacitance being switched per clock cycle (proportional to the number of transistors whose inputs change), V is voltage, and F is the processor frequency (cycles per second). Increases in frequency increase the amount of power used in a processor. Increasing processor power consumption led ultimately to Intel's May 8, 2004 cancellation of its Tejas and Jayhawk processors, which is generally cited as the end of frequency scaling as the dominant computer architecture paradigm. Moore's law is the empirical observation that the number of transistors in a microprocessor doubles every 18 to 24 months. Despite power consumption issues, and repeated predictions of its end, Moore's law is still in effect. With the end of frequency scaling, these additional transistors (which are no longer used for frequency scaling) can be used to add extra hardware for parallel computing.";
        String summarizableText="TO ENSURE that it meets the 750 new rules on capital imposed in the aftermath of the financial crisis, JPMorgan Chase employs over 950 people. A further 400 or so try to follow around 500 regulations on the liquidity of its assets, designed to stop the bank toppling over if markets seize up. A team of 300 is needed to monitor compliance with the Volcker rule, which in almost 1,000 pages restricts banks from trading on their own account.The intention of all these rules is to prevent a repeat of the bankruptcies and bail-outs of 2008. But some observers, including JPMorgan’s boss, Jamie Dimon, and Larry Summers, a former Treasury secretary, argue that in their rush to make banks safer, regulators may have created a riskier financial system. By throttling the bits of banks that “make markets” in bonds, shares, currencies and commodities, the theory goes, watchdogs have made such assets less liquid. Investors may not be able to buy and sell them quickly, cheaply and without moving the price. The consequences in a downturn, when markets are less liquid anyway, could be severe. Banks have undoubtedly cut back as the plethora of new rules has made it difficult for their trading arms to eke out a satisfactory profit. They used to “warehouse” lots of bonds and other securities they had bought from one client and hoped to sell to another. But they must now hold more capital and liquid assets to offset the potential losses from trading, so keep much smaller inventories and place fewer bets. Broadly speaking, trading desks are still happy to match buyers and sellers but are reluctant to commit to a purchase before lining up a buyer. Meanwhile, the value of outstanding bonds has swollen to record levels, most of them in the hands of asset managers (see chart). That is in part a corollary of banks trimming lending, and so pushing borrowers to the bond market instead, and in part a natural response to low interest rates. Even firms with patchy credit records are issuing “high-yield” debt to investors clamouring for returns. Governments have remained eager borrowers, too. The result is an imbalance. In America, investment funds used to hold only three times as many bonds as banks. Now they hold 20 times as many, according to the Federal Reserve (see chart). Mr Dimon paints an even starker picture for Treasuries. In 2007 JPMorgan and its peers used to have $2.7 trillion available to make markets. Now they have just $1.7 trillion—while the American national debt has doubled. In Europe, where banks have trimmed investment banking even more, the situation is if anything worse. The result of this lopsidedness, pessimists say, are events like the “flash crash” last year, during which yields on Treasuries suddenly tumbled by 0.34 percentage points for no apparent reason—an extraordinary shift for the bedrock security of the global financial system. They are worried about bonds of all sorts, which are much less heavily traded than shares, currencies and commodities. Funds that track corporate bonds often promise their investors their money back whenever they want it, despite the relative illiquidity of their assets. The IMF recently calculated that it might take 50-60 days for a fund holding American high-yield corporate bonds to find buyers for its securities. Meanwhile, investors are typically entitled to their money back within seven days of asking for it. “No investment vehicle should promise greater liquidity than is afforded by its underlying assets,” says Howard Marks, boss of Oaktree, a debt fund. Regulators are mindful of all this. The Securities and Exchange Commission in America has called for stress tests of asset managers to ensure they can muddle through a crisis. The Bank of England wants them to look closely at redemption policies. They also suspect, however, that the high level of liquidity before the crisis was an anomaly that bankers are harping on about in an effort to roll back regulation. Asset managers are also aware of the risks of diminished liquidity. BlackRock, the world’s biggest, has said it is limiting its exposure to certain bonds as a result. Others are breaking up big trades into smaller orders, to prevent them moving prices in an adverse direction, or trading less than they might otherwise. Funds tracking bond indices hold cash to meet redemptions. They can also invest in derivatives linked to the index, which are typically more liquid than individual bonds. If faced with a rash of redemptions, these can be sold off without much loss. Another solution is for the asset managers to bypass the banks. Many are trying to “cross-trade”, exchanging assets with one another directly, instead of using banks as go-betweens. But matching buy and sell orders electronically is tricky for bonds: whereas most firms have only one or two classes of shares, many have issued dozens of bonds, in different currencies and with different maturities. There have been several attempts to set up trading platforms, but few have attracted much volume. Even if such schemes get off the ground, asset managers cannot fully substitute for banks. They do not have as much purchasing power, since their balance-sheets are not swollen with borrowed money. Relatively few of them have a mandate to be contrarian: most (especially those passively tracking an index) want to enter or exit the same positions at the same time. All this may mean that asset managers are indeed forced to offload securities at fire-sale prices in times of turmoil. But unlike banks, which can fail due to trading losses, asset managers are mere custodians of money. Any losses in their funds are passed on directly to investors. Having banks—highly leveraged and interconnected institutions—sit on top of that risk proved a disastrous recipe during the crisis. Maybe their retrenchment has indeed made markets riskier. Yet that may be an acceptable price for making banks safer.";
        //String summarizableText="SUGAR and spice motivated many an explorer , and the voyages of discovery that resulted from European demand for these products were the basis for building powerful empires . Today , the same resources are still stimulating the development of new trading societies - - but now those societies are growing inside computers , rather than overseas . And in watching these artificial societies grow , their inventors are starting their own voyage of discovery - - one they hope will provide insights into real societies that have thus far been denied to conventional social science . Two of these inventors are Robert Axtell and Joshua Epstein , who work at the Brookings Institution in Washington , DC . They have created an artificial world they call ' Sugar scape ` . In it , software ' agents ` of their devising live out their lives . The agents devised by Dr Axtell and Dr Epstein appear on a computer screen as little red dots which move over a 50 by 50 grid . Each is actually a piece of software running inside the computer . But , like people in a real society , the Sugarscape agents are not identical . They have different visual abilities and different demands for sugar . The computer landscape has two mountains of a resource , called ' sugar ` , which the agents require . All of them follow the same rules: look around , go to the unoccupied spot with the largest amount of sugar , and then eat the sugar . The sugar , when consumed , grows back at a pre - determined rate . When only sugar is present , the agents ' behaviour is boringly predictable . Most of them cluster in the sugar mountains . Only a few of the extremely short - sighted are left out in the cold . But , having proved their model worked , Dr Axtell and Dr Epstein went on to add complexity to it in the form of a second resource - - ' spice ` . In this version , the agents require both sugar and spice , but they do not necessarily have to gather their needs directly . Instead , they may trade with one another . Each time a trade happens , the computer registers the price of one good in terms of the other . In the most simple sugar and spice models , the market settles on one price for each commodity . It does so with no central planning - - a happy story for orthodox economics . However , as Dr Axtell and Dr Epstein added further complexity to their model the story became messier . The simple model had agents who never died and whose preferences never changed . But if old agents are allowed to pass away and be replaced by new ones , and if agents ' preferences for sugar and spice evolve as they interact , the cyber - market never settles on a single price ( just like real life ) . Sugarscape is not the only agent - based model that has started to explore the pattern of trade . The agents in a model devised by Deborah Duong ( who works at George Mason University , in Fairfax , Virginia ) , can produce a variety of goods ( she calls them ' oats ` , ' peas ` , ' beans ` and ' barley ` ) . They , too , can trade . But they can also vary the amount of each good they produce . In Ms Duong 's model , each agent quickly learns to tell which other agents sell the produce needed to complement its own diet . It then adjusts its behaviour accordingly . When she ran her model , Ms Duong found that it vindicated Adam Smith . It ended up with a division of labour - - although each agent can ' grow ` the full range of goods if it chooses to , it generally decides that it is more efficient to produce just one and trade it for the others . Ms Duong 's agents , then , spontaneously organise themselves into a barter economy . Indeed , they sometimes go further , with one of the goods taking on a money - like quality and being used as a standard of trade . The next step is to build models that can be tested against the real world . In one such test Dr Axtell and Dr Epstein have collaborated with George Gumerman , an archaeologist from the Santa Fe Institute in New Mexico , to model the Anasazi society . This was a culture that flourished in the American Southwest for hundreds of years , built astonishing cliff dwellings , and then vanished . But the simulation has not yet found the reason why it did so . Rather than disappearing , computer - based versions of Anasazi society sometimes go on . And on , and on , and on , and on . . . ";
        //String summarizableText="WASHINGTON :  In an announcement that could forever change the way scientists study the hydrogen-based star, NASA researchers published a comprehensive study today theorizing that the sun may be capable of supporting fire-based lifeforms. \"After extensive research, we have reason to believe that the sun may be habitable for fire-based life, including primitive single-flame microbes and more complex ember-like organisms capable of thriving under all manner of burning conditions\", lead investigator Dr.Steven T Aukerman wrote, noting that the sun’s helium-rich surface of highly charged particles provides the perfect food source for fire-based lifeforms. With a surface temperature of 10,000 degrees Fahrenheit and frequent eruptions of ionized gases flowing along strong magnetic fields, the sun is the first star we’ve seen with the right conditions to support fire organisms, and we believe there is evidence to support the theory that fire-bacteria, fire-insects, and even tiny fire-fish were once perhaps populous on the sun’s surface. Scientists cautioned that despite the exciting possibilities of fire-life on the star, there are numerous logistical, moral, and ethical questions to resolve before scientists could even begin to entertain the possibility of putting fire-people on the sun.";

        //System.out.println(summarizableText);
        System.out.println();
		if(!summarizableText.equals("")){
			
			preprocessedText = summarizableText;
			
		}
		else{
			throw new Exception(Constants.TEXT_EXTRACTION_FAILED);
		}
		
		extractSentence = new ExtractSentence();
		sentences = extractSentence.getSentences(preprocessedText);
        int count;
        for(String s : sentences)
        {
            //System.out.println(s);
        
        }
        count=sentences.size();
        //System.out.println();
                //System.out.println();

       // System.out.println();

       // System.out.println();

        
       double[][] similarityMatrix = getSimilarityMatrixFromSentenceList(sentences);
        
        /*for(int i=0;i<similarityMatrix.length;i++)
        {
            for(int j=0;j<similarityMatrix.length;j++)
                {
                System.out.print("   "+similarityMatrix[i][j]);
                }
            System.out.println();
        }
*/
    
                                    
        
		lexrank = new Lexrank(sentences, similarityMatrix	,0.85, degree);
		double[] resultEigenValues = lexrank.powerMethod(.001);
		
		//printArray(resultEigenValues);
		
        //System.out.println(summary);
               // System.out.println();
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
                    //String temp1=sentences[j];
                    //sentences[j]=sentences[j+1];
                    //sentences[j+1]=temp1;
                    Collections.swap(sentences,j,j+1);
                }
        
        System.out.println();
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
        
		//System.out.println(dvc.generateTF_IDFScores().toString());
		score = dvc.generateTF_IDFScores();
		//System.out.println(score);
        
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
                        //System.out.print("    "+result[i][j]);

                    }
                    else
                    {
                        result[i][j]=0;
                        //System.out.print("    "+result[i][j]);

                    }
				}

				//System.out.println(" ");
			}
			
		
        //for(int i=0;i<degree.length;i++)
            //System.out.print(" "+degree[i]);
        //System.out.println();
		return result;
		
	}
	public static void main(String...a) throws Exception{
        System.out.println();
        System.out.println();

		Controller c = new Controller();
        // c.getSummary();
        System.out.println(c.getSummary());
        System.out.println();
        
	}


}
