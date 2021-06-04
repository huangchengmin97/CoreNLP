package edu.stanford.nlp.parser.lexparser;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.trees.DiskTreebank;
import edu.stanford.nlp.trees.HeadFinder;
import edu.stanford.nlp.trees.LeftHeadFinder;
import edu.stanford.nlp.trees.MemoryTreebank;
import edu.stanford.nlp.trees.PennTreeReaderFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeReaderFactory;
import edu.stanford.nlp.trees.TreeTransformer;
import edu.stanford.nlp.trees.international.hungarian.HungarianTreebankLanguagePack;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * Bare-bones implementation of a ParserParams for the Hungarian SPMRL treebank.
 * <br>
 * Suitable for use in the SR Parser.  May need additional work to function in the PCFG.
 * Also, would likely function better with a new headfinder.
 */
public class HungarianTreebankParserParams extends AbstractTreebankParserParams  {
  /** A logger for this class */
  private static final Redwood.RedwoodChannels log = Redwood.channels(HungarianTreebankParserParams.class);

  public HungarianTreebankParserParams() {
    super(new HungarianTreebankLanguagePack());
    // TODO: make a Hungarian specific HeadFinder or build one that can be learned
    headFinder = new LeftHeadFinder();
  }

  private HeadFinder headFinder;

  static final String[] EMPTY_SISTERS = new String[0];

  @Override
  public HeadFinder headFinder() {
    return headFinder;
  }

  @Override
  public HeadFinder typedDependencyHeadFinder() {
    return headFinder;
  }

  /**
   * Allows you to read in trees from the source you want.  It's the
   * responsibility of treeReaderFactory() to deal properly with character-set
   * encoding of the input.  It also is the responsibility of tr to properly
   * normalize trees.
   */
  @Override
  public DiskTreebank diskTreebank() {
    return new DiskTreebank(treeReaderFactory());
  }


  /**
   * Allows you to read in trees from the source you want.  It's the
   * responsibility of treeReaderFactory() to deal properly with character-set
   * encoding of the input.  It also is the responsibility of tr to properly
   * normalize trees.
   */
  @Override
  public MemoryTreebank memoryTreebank() {
    return new MemoryTreebank(treeReaderFactory());
  }

  @Override
  public TreeTransformer collinizer() {
    return new TreeCollinizer(tlp, true, false, 0);
  }

  @Override
  public TreeTransformer collinizerEvalb() {
    return collinizer();
  }

  @Override
  public String[] sisterSplitters() {
    // TODO: no idea if this can/should be improved
    return EMPTY_SISTERS;
  }

  @Override
  public Tree transformTree(Tree t, Tree root) {
    // TODO
    return t;
  }

  /** {@inheritDoc} */
  @Override
  public TreeReaderFactory treeReaderFactory() {
    return new PennTreeReaderFactory();
  }


  @Override
  public void display() {
    String hungarianParams = "Using HungarianTreebankParserParams";
    log.info(hungarianParams);
  }

  /** {@inheritDoc} */
  @Override
  public List<? extends HasWord> defaultTestSentence() {
    List<Word> ret = new ArrayList<>();
    String[] sent = {"Ez", "egy", "teszt", "."};
    for (String str : sent) {
      ret.add(new Word(str));
    }
    return ret;
  }
}
