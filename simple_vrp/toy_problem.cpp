
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>
#include <queue>
#include <cmath>
#include <time.h>

#include <cassert>
#include <iostream>

struct State
{
  std::array<double, 2> p; // position
  double finish_time = 0.0;
  double travel_distance = 0.0;

  static State make(std::array<double, 2> p_)
  {
    return State{p_};
  }
};

//========================================================================

class TaskRequest;
using ConstTaskRequestPtr = std::shared_ptr<TaskRequest>;

class Candidates
{
public:

  struct Entry
  {
    std::size_t candidate;
    State state;
  };

  using Map = std::multimap<double, Entry>;

  static Candidates make(
      const std::vector<State>& initial_states,
      const TaskRequest& request);

  Candidates(const Candidates& other)
  {
    _value_map = other._value_map;
    _update_map();
  }

  Candidates& operator=(const Candidates& other)
  {
    _value_map = other._value_map;
    _update_map();
    return *this;
  }

  Candidates(Candidates&&) = default;
  Candidates& operator=(Candidates&&) = default;

  struct Range
  {
    Map::const_iterator begin;
    Map::const_iterator end;
  };

  Range best_candidates() const
  {
    assert(!_value_map.empty());

    Range range;
    range.begin = _value_map.begin();
    auto it = range.begin;
    while (it->first == range.begin->first)
      ++it;

    range.end = it;
    return range;
  }

  double best_finish_time() const
  {
    assert(!_value_map.empty());
    return _value_map.begin()->first;
  }

  void update_candidate(std::size_t candidate, State state)
  {
    const auto it = _candidate_map.at(candidate);
    _value_map.erase(it);
    _candidate_map[candidate] = _value_map.insert(
      {state.finish_time, Entry{candidate, state}});
  }


private:
  Map _value_map;
  std::vector<Map::iterator> _candidate_map;

  Candidates(Map candidate_values)
    : _value_map(std::move(candidate_values))
  {
    _update_map();
  }

  void _update_map()
  {
    for (auto it = _value_map.begin(); it != _value_map.end(); ++it)
    {
      const auto c = it->second.candidate;
      if (_candidate_map.size() <= c)
        _candidate_map.resize(c+1);

      _candidate_map[c] = it;
    }
  }
};

//========================================================================

struct PendingTask
{
  PendingTask(
      std::vector<State> initial_states,
      ConstTaskRequestPtr request_)
    : request(std::move(request_)),
      candidates(Candidates::make(initial_states, *request))
  {
    // Do nothing
  }

  ConstTaskRequestPtr request;
  Candidates candidates;
};

struct Assignment
{
  std::size_t task_id;
  State state;
};

struct Node
{
  std::vector<std::vector<Assignment>> assignments;
  std::unordered_map<std::size_t, PendingTask> unassigned;
  double cost_estimate;
};

using NodePtr = std::shared_ptr<Node>;
using ConstNodePtr = std::shared_ptr<const Node>;

class TaskRequest
{
public:

  virtual State estimate(const State& initial_state) const = 0;

};

void print_node(const Node& node)
{
  std::cout << " -- " << node.cost_estimate << ": <";
  for (std::size_t a=0; a < node.assignments.size(); ++a)
  {
    if (a > 0)
      std::cout << ", ";

    std::cout << a << ": [";
    for (const auto i : node.assignments[a])
      std::cout << " " << i.task_id;
    std::cout << " ]";
  }

  std::cout << " -- ";
  bool first = true;
  for (const auto& u : node.unassigned)
  {
    if (first)
      first = false;
    else
      std::cout << ", ";

    std::cout << u.first << ":";
    const auto& range = u.second.candidates.best_candidates();
    for (auto it = range.begin; it != range.end; ++it)
      std::cout << " " << it->second.candidate;
  }

  std::cout << ">" << std::endl;
}

Candidates Candidates::make(
    const std::vector<State>& initial_states,
    const TaskRequest& request)
{
  Map initial_map;
  for (std::size_t s=0; s < initial_states.size(); ++s)
  {
    const auto& state = initial_states[s];
    const auto finish = request.estimate(state);
    initial_map.insert({finish.finish_time, Entry{s, finish}});
  }

  return Candidates(std::move(initial_map));
}

//========================================================================

double g(const Node& n, bool details)
{
  double output = 0.0;
  details = true;

  std::size_t a = 0;
  for (const auto& agent : n.assignments)
  {
    if (details)
      std::cout << "Costs for agent " << a++ << std::endl;

    for (const auto& assignment : agent)
    {
      output += assignment.state.finish_time;

      if (details)
      {
        std::cout << " -- " << output << " <- " << assignment.state.finish_time
                  << std::endl;
      }
    }
  }

  return output;
}

double h(const Node& n, bool details = false)
{
  double output = 0.0;

  if (details)
    std::cout << "Unassigned costs" << std::endl;

  for (const auto& u : n.unassigned)
  {
    output += u.second.candidates.best_finish_time();

    if (details)
    {
      std::cout << " -- " << output << " <- "
                << u.second.candidates.best_finish_time() << std::endl;
    }
  }

  return output;
}

double f(const Node& n, bool details = false)
{
  return g(n, details) + h(n, details);
}

//========================================================================

class Filter
{
public:

  Filter(bool passthrough)
    : _passthrough(passthrough)
  {
    // Do nothing
  }

  bool ignore(const Node& node);

private:

  struct TaskTable;

  struct AgentTable
  {
    std::unordered_map<std::size_t, std::unique_ptr<TaskTable>> agent;
  };

  struct TaskTable
  {
    std::unordered_map<std::size_t, std::unique_ptr<AgentTable>> task;
  };

  bool _passthrough;
  AgentTable _root;
};

bool Filter::ignore(const Node& node)
{
  if (_passthrough)
    return false;

  bool new_node = false;

  // TODO(MXG): Consider replacing this tree structure with a hash set

  AgentTable* agent_table = &_root;
  std::size_t a = 0;
  std::size_t t = 0;
  while(a < node.assignments.size())
  {
    const auto& current_agent = node.assignments.at(a);

    if (t < current_agent.size())
    {
      const auto& task_id = current_agent[t].task_id;
      const auto agent_insertion = agent_table->agent.insert({a, nullptr});
      if (agent_insertion.second)
        agent_insertion.first->second = std::make_unique<TaskTable>();

      auto* task_table = agent_insertion.first->second.get();

      const auto task_insertion = task_table->task.insert({task_id, nullptr});
      if (task_insertion.second)
      {
        new_node = true;
        task_insertion.first->second = std::make_unique<AgentTable>();
      }

      agent_table = task_insertion.first->second.get();
      ++t;
    }
    else
    {
      t = 0;
      ++a;
    }
  }

  return !new_node;
}

//========================================================================

std::vector<ConstNodePtr> expand(ConstNodePtr parent, Filter& filter)
{
  std::vector<ConstNodePtr> new_nodes;
  new_nodes.reserve(parent->unassigned.size());
  for (const auto& u : parent->unassigned)
  {
    const auto& range = u.second.candidates.best_candidates();
    for (auto it = range.begin; it != range.end; ++it)
    {
      const auto& c = it->second;

      auto new_node = std::make_shared<Node>(*parent);
      new_node->assignments[c.candidate]
          .push_back(Assignment{u.first, c.state});

      new_node->unassigned.erase(u.first);

      for (auto& new_u : new_node->unassigned)
      {
        new_u.second.candidates.update_candidate(
              c.candidate, new_u.second.request->estimate(c.state));
      }

      new_node->cost_estimate = f(*new_node);

      if (filter.ignore(*new_node))
      {
        std::cout << "ignoring: ";
        print_node(*new_node);
        continue;
      }

      new_nodes.push_back(std::move(new_node));
    }
  }

  return new_nodes;
}

struct LowestCostEstimate
{
  bool operator()(const ConstNodePtr& a, const ConstNodePtr& b)
  {
    return b->cost_estimate < a->cost_estimate;
  }
};

ConstNodePtr solve(
    std::vector<State> initial_states,
    std::vector<ConstTaskRequestPtr> requests,
    const bool use_filter,
    const bool display = false)
{
  auto initial_node = std::make_shared<Node>();

  initial_node->assignments.resize(initial_states.size());
  for (std::size_t i=0; i < requests.size(); ++i)
  {
    const auto r = requests[i];
    initial_node->unassigned.insert({i, PendingTask(initial_states, r)});
    initial_node->cost_estimate = f(*initial_node);
  }

  using Queue = std::priority_queue<
      ConstNodePtr,
      std::vector<ConstNodePtr>,
      LowestCostEstimate>;

  Queue queue;
  queue.push(initial_node);

  Filter filter(!use_filter);

  std::size_t total_queue_entries = 1;
  std::size_t total_queue_expansions = 0;
  while (!queue.empty())
  {
    if (display)
    {
      auto display_queue = queue;
      while (!display_queue.empty())
      {
        const auto top = display_queue.top();
        display_queue.pop();
        print_node(*top);
      }

      std::cout << "=========================" << std::endl;
    }

    const auto top = queue.top();
    queue.pop();

    if (top->unassigned.empty())
    {
      std::cout << "Winning cost: " << top->cost_estimate << std::endl;
      std::cout << "Final queue size: " << queue.size() << std::endl;
      std::cout << "Total queue expansions: " << total_queue_expansions << std::endl;
      std::cout << "Total queue entries: " << total_queue_entries << std::endl;

      // This is the solution criteria
      return top;
    }

    // We don't have a solution yet, so let's expand
    const auto new_nodes = expand(top, filter);
    ++total_queue_expansions;
    total_queue_entries += new_nodes.size();

    for (const auto& n : new_nodes)
      queue.push(n);
  }

  return nullptr;
}

//========================================================================

class TravelTaskRequest : public TaskRequest
{
public:

  TravelTaskRequest(std::array<double, 2> p, double speed = 1.0)
    : _p(p),
      _speed(speed)
  {
    // Do nothing
  }

  static ConstTaskRequestPtr make(std::array<double, 2> p, double speed = 1.0)
  {
    return std::make_shared<TravelTaskRequest>(p, speed);
  }

  State estimate(const State& initial_state) const final
  {
    State output;
    output.p = _p;

    const auto& p0 = initial_state.p;
    const double dx = _p[0] - p0[0];
    const double dy = _p[1] - p0[1];
    const double dist = std::sqrt(dx*dx + dy*dy);

    output.finish_time = initial_state.finish_time + dist/_speed;
    output.travel_distance = dist;
    return output;
  }

private:
  std::array<double, 2> _p;
  double _speed;
};

//========================================================================

int main()
{
  std::vector<ConstTaskRequestPtr> requests =
  {
    TravelTaskRequest::make({5, 5}),    // 0
    TravelTaskRequest::make({-5, 5}),   // 1
    TravelTaskRequest::make({10, -10}), // 2    
    TravelTaskRequest::make({1, 0}),    // 3
    TravelTaskRequest::make({2, -30}),    // 4
    TravelTaskRequest::make({-2, 0}),    // 5 
    TravelTaskRequest::make({-7, -1}),    // 
    TravelTaskRequest::make({6, 2})    // 
  };

  std::vector<State> initial_states =
  {
    State::make({0, 0}),
    State::make({5, 0}),
    State::make({-3, 4})
  };

  clock_t tStart = clock();

  const auto solution = solve(initial_states, requests, true, true);

  if (!solution)
  {
    std::cout << "No solution found! :(" << std::endl;
    return 0;
  }

  std::cout << "Assignments:\n";
  for (std::size_t i=0; i < solution->assignments.size(); ++i)
  {
    std::cout << i << ":";
    for (const auto& t : solution->assignments[i])
      std::cout << "  " << t.task_id;
    std::cout << "\n";
  }

  printf("Time taken: %.4fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}


// ========================================================
// My own planning

// using TaskRequest 

// using RobotTaskQueue = std::queue<TaskRequest>

// std::vector<std::vector<Assignment>> assignments;
