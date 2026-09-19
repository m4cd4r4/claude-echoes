// Day windows for the temporal eval cases, derived from the gold message's age.
//
// A fixture that stores a literal `days` number ages out: three weeks after the
// set is built, every include window has slid past its gold and every exclude
// window trivially passes. The eval therefore recomputes the window at run time
// from the gold row's created_at, and the builder uses the same function so the
// two can never disagree.
export const ageInDays = (createdAt, now = Date.now()) =>
  Math.max(1, Math.ceil((now - new Date(createdAt)) / 86400000));

// include: the window reaches 2 days past the gold, so it must still be found.
// exclude: the window covers only the most recent third of that age, so the
// gold predates it and must NOT be returned.
export const temporalDays = (category, ageDays) =>
  category === 'temporal_exclude' ? Math.max(1, Math.floor(ageDays / 3)) : ageDays + 2;
