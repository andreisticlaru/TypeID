// Casual, text-message-style sentences used as random transcription prompts.
// Kept lowercase/informal on purpose — closer to natural typing than literary prose.
const SENTENCES = [
  "hey are we still on for tonight or did that get moved",
  "omg i just saw the funniest thing on my walk home",
  "can you grab milk on your way back, we're almost out",
  "not gonna lie today has been a lot lol",
  "just landed, my phone's at 3% so ill call you later",
  "wait did you already eat or should i order something",
  "i think i left my charger at your place again",
  "running like ten minutes late, so sorry",
  "the show was actually way better than i expected",
  "lowkey considering just staying in this weekend",
  "did you see the email from the landlord about parking",
  "no worries, take your time, im not in a rush",
  "i keep meaning to text you back and then i forget",
  "the wifi here is so bad i can barely load anything",
  "we should just cancel and reschedule for next week",
  "honestly i don't even remember what we argued about",
  "can you send me that photo from saturday",
  "i just woke up and i already regret staying up so late",
  "meet you outside in five, im parking now",
  "this coffee is way too strong but im drinking it anyway",
  "i think the package finally shipped, tracking says tomorrow",
  "sorry for the late reply, work has been insane",
  "are you free to talk or is this a bad time",
  "i cant believe its already almost the end of the month",
  "lets just figure it out when we get there",
  "my phone autocorrected that into something ridiculous",
  "i left the oven on and had to turn around and go back",
  "the meeting got pushed to friday, ill send the new invite",
  "i think i liked the old version better honestly",
  "give me two minutes, im just finishing something up",
];

function pickSentence(exclude) {
  const pool = exclude
    ? SENTENCES.filter((s) => s !== exclude)
    : SENTENCES;
  return pool[Math.floor(Math.random() * pool.length)];
}
