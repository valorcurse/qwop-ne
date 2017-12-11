//this will hold the new population of genomes
vector<CGenome> NewPop;
//request the offspring from each species. The number of children to
//spawn is a double which we need to convert to an int.
int NumSpawnedSoFar = 0;
CGenome baby;
//now to iterate through each species selecting offspring to be mated and
//mutated
for (int spc=0; spc<m_vecSpecies.size(); ++spc)
{
//because of the number to spawn from each species is a double
//rounded up or down to an integer it is possible to get an overflow
//of genomes spawned. This statement just makes sure that doesn't
//happen
	if (NumSpawnedSoFar < CParams::iNumSweepers)
	{
//this is the amount of offspring this species is required to
// spawn. Rounded simply rounds the double up or down.
		int NumToSpawn = Rounded(m_vecSpecies[spc].NumToSpawn());
		bool bChosenBestYet = false;
		while (NumToSpawn--)
		{
//first grab the best performing genome from this species and transfer
//to the new population without mutation. This provides per species
//elitism
			if (!bChosenBestYet)
			{
				baby = m_vecSpecies[spc].Leader();
				bChosenBestYet = true;
			}
			else
			{
//if the number of individuals in this species is only one
//then we can only perform mutation
				if (m_vecSpecies[spc].NumMembers() == 1)
				{
//spawn a child
					baby = m_vecSpecies[spc].Spawn();
				}
//if greater than one we can use the crossover operator
				else
				{
//spawn1
					CGenome g1 = m_vecSpecies[spc].Spawn();
					if (RandFloat() < CParams::dCrossoverRate)
					{
//spawn2, make sure it's not the same as g1
						CGenome g2 = m_vecSpecies[spc].Spawn();
// number of attempts at finding a different genome
						int NumAttempts = 5;
						while ( (g1.ID() == g2.ID()) && (NumAttempts--) )
						{
							g2 = m_vecSpecies[spc].Spawn();
						}
						if (g1.ID() != g2.ID())
						{
							baby = Crossover(g1, g2);
						}
					}
					else
					{
						baby = g1;
					}
				}
				++m_iNextGenomeID;
				baby.SetID(m_iNextGenomeID);
//now we have a spawned child lets mutate it! First there is the
//chance a neuron may be added
				if (baby.NumNeurons() < CParams::iMaxPermittedNeurons)
				{
					baby.AddNeuron(CParams::dChanceAddNode,
						*m_pInnovation,
						CParams::iNumTrysToFindOldLink);
				}
//now there's the chance a link may be added
				baby.AddLink(CParams::dChanceAddLink,
					CParams::dChanceAddRecurrentLink,
					*m_pInnovation,
					CParams::iNumTrysToFindLoopedLink,
					CParams::iNumAddLinkAttempts);
//mutate the weights
				baby.MutateWeights(CParams::dMutationRate,
					CParams::dProbabilityWeightReplaced,
					CParams::dMaxWeightPerturbation);
//mutate the activation response
				baby.MutateActivationResponse(CParams::dActivationMutationRate,
					CParams::dMaxActivationPerturbation);
			}
//sort the babies genes by their innovation numbers
			baby.SortGenes();
//add to new pop
			NewPop.push_back(baby);
			++NumSpawnedSoFar;
			if (NumSpawnedSoFar == CParams::iNumSweepers)
			{
				NumToSpawn = 0;
			}
}//end while
}//end if
}//next species
//if there is an underflow due to a rounding error when adding up all
//the species spawn amounts, and the amount of offspring falls short of
//the population size, additional children need to be created and added
//to the new population. This is achieved simply, by using tournament
//selection over the entire population.
if (NumSpawnedSoFar < CParams::iNumSweepers)
{
//calculate the amount of additional children required
	int Rqd = CParams::iNumSweepers - NumSpawnedSoFar;
//grab them
	while (Rqd--)
	{
		NewPop.push_back(TournamentSelection(m_iPopSize/5));
	}
}
//replace the current population with the new one
m_vecGenomes = NewPop;
//create the new phenotypes
vector<CNeuralNet*> new_phenotypes;
for (gen=0; gen<m_vecGenomes.size(); ++gen)
{
//calculate max network depth
	int depth = CalculateNetDepth(m_vecGenomes[gen]);
	CNeuralNet* phenotype = m_vecGenomes[gen].CreatePhenotype(depth);
	new_phenotypes.push_back(phenotype);
}
//increase generation counter
++m_iGeneration;
return new_phenotypes;
}
