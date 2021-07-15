package edu.stanford.nlp.parser.metrics;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.common.ParserQuery;
import edu.stanford.nlp.parser.common.ParserQueryFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.ScoredObject;

/**
 * Memorize a bunch of parser results provided by an outside parser, then
 * return them as requested.  Useful for running our eval code on an
 * external parser
 *
 * @author John Bauer
 */
public class ExternalParser implements ParserQueryFactory {
  Map<List<String>, List<ScoredObject<Tree>>> results;

  public ExternalParser(Map<List<String>, List<ScoredObject<Tree>>> results) {
    this.results = new HashMap<>(results);
  }

  @Override
  public ParserQuery parserQuery() {
    return new ExternalParserQuery(this);
  }
}
