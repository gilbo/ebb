/* Copyright 2015 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <map>
#include <set>

#include "default_mapper.h"
#include "serialize.h"

#include "ebb_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

// legion logger
static LegionRuntime::Logger::Category log_mapper("mapper");

// message types
enum MapperMessageType {
  MAPPER_RECORD_FIELD,
  MAPPER_TOTAL_MESSAGES
};

class EbbMapper : public DefaultMapper {
public:
  EbbMapper(Machine machine, Processor local,
            const char *mapper_name = NULL);
  virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);
  virtual void default_policy_select_constraint_fields(
                                    MapperContext ctx,
                                    const RegionRequirement &req,
                                    std::vector<FieldID> &fields);
  virtual bool default_policy_select_close_virtual(const MapperContext ctx,
                                                   const Close &close);
  virtual void handle_message(const MapperContext ctx,
                              const MapperMessage& message);
private:
  // active fields for a logical region
  std::map<std::string, std::set<FieldID> >   active_fields;
  std::vector<Processor>                      all_procs;
  int                                         n_nodes;
  int                                         per_node;
  // get logical region corresponding to a region requirement
  LogicalRegion get_root_region(MapperContext ctx,
                                   const RegionRequirement &req);
  LogicalRegion get_root_region(MapperContext ctx,
                                const LogicalRegion &handle);
  LogicalRegion get_root_region(MapperContext ctx,
                                const LogicalPartition &handle);
};  // class EbbMapper

// EbbMapper constructor
EbbMapper::EbbMapper(Machine machine, Processor local,
                     const char *mapper_name)
  : DefaultMapper(machine, local, mapper_name)
{
  Machine::ProcessorQuery query_all_procs =
        Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC);
  all_procs.insert(all_procs.begin(),
                   query_all_procs.begin(),
                   query_all_procs.end());
  n_nodes = 0;
  for(int i=0; i<all_procs.size(); i++) {
    if (all_procs[i].address_space() >= n_nodes) {
      n_nodes = all_procs[i].address_space() + 1;
    }
  }
  per_node = all_procs.size() / n_nodes;

  //if (local.id == all_procs[0].id) {
  //  printf("Hello from mapper\n");
  //  for(int i=0; i<all_procs.size(); i++)
  //    printf(" PROC %d %llx %d %u\n", i, all_procs[i].id,
  //                                 all_procs[i].kind(),
  //                                 all_procs[i].address_space());
  //}
}

Processor EbbMapper::default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task) {
  if (false && task.tag != 0) {
    // all_procs here is a safety modulus
    int proc_num = (task.tag - 1)%all_procs.size();
    Processor p = all_procs[proc_num];
    //printf("Launching Tagged on %llx %lx\n", p.id, task.tag);
    return p;
  } else if (!task.regions.empty() &&
             task.regions[0].handle_type == SINGULAR)
  {
    Color index = mapper_rt_get_logical_region_color(ctx, task.regions[0].region);
    int proc_off = (int(index) / n_nodes)%per_node;
    int node_off = int(index) % n_nodes;
    // all_procs here is a safety modulus
    int proc_num = (node_off*per_node + proc_off)%all_procs.size();
    Processor p = all_procs[proc_num];
    //printf("Launching Tagless on %llx %lx\n", p.id, task.tag);
    return p;
  }
  return DefaultMapper::default_policy_select_initial_processor(ctx, task);
}

void EbbMapper::default_policy_select_constraint_fields(
                                    MapperContext ctx,
                                    const RegionRequirement &req,
                                    std::vector<FieldID> &fields) {
  LogicalRegion root = get_root_region(ctx, req);
  const char *name_c;
  mapper_rt_retrieve_name(ctx, root, name_c);
  std::string name(name_c);

  std::set<FieldID> &additional = active_fields[name];
  for (std::set<FieldID>::iterator pf_it = req.privilege_fields.begin();
       pf_it != req.privilege_fields.end(); pf_it++) {
    FieldID pf = *pf_it;
    if (additional.find(pf) == additional.end()) {
      // broadcast and record a new field
      Realm::Serialization::DynamicBufferSerializer buffer(name.size() + 16);
      buffer << (int)MAPPER_RECORD_FIELD;
      buffer << name;
      buffer << req.privilege_fields;
      // broadcast_message(buffer.get_buffer(), buffer.bytes_used());
      additional.insert(pf);
    }
  }
  fields.insert(fields.begin(), additional.begin(), additional.end());
}

bool EbbMapper::default_policy_select_close_virtual(const MapperContext ctx,
                                                    const Close &close)
{
  return false;
}

void EbbMapper::handle_message(const MapperContext ctx,
                               const MapperMessage& message)
{
  Realm::Serialization::FixedBufferDeserializer buffer(message.message, message.size);
  int msg_type;
  buffer >> msg_type;
  switch(msg_type) {
    case MAPPER_RECORD_FIELD:
    {
      std::string name;
      buffer >> name;
      std::set<FieldID> privilege_fields;
      buffer >> privilege_fields;
      active_fields[name].insert(privilege_fields.begin(),
                                 privilege_fields.end());
      break;
    }
    default:
    {
      printf("Invalid message recieved by mapper\n");
      assert(false);
    }
  }
}

LogicalRegion EbbMapper::get_root_region(MapperContext ctx,
                                            const RegionRequirement &req) {
  LogicalRegion root;
  if (req.handle_type == SINGULAR || req.handle_type == REG_PROJECTION) {
    root = get_root_region(ctx, req.region);
  } else {
    assert(req.handle_type == PART_PROJECTION);
    root = get_root_region(ctx, req.partition);
  }
  return root;
}

LogicalRegion EbbMapper::get_root_region(MapperContext ctx,
                                         const LogicalRegion &handle) {
  if (mapper_rt_has_parent_logical_partition(ctx, handle)) {
    return get_root_region(ctx, mapper_rt_get_parent_logical_partition(ctx, handle));
  }
  return handle;
}

LogicalRegion EbbMapper::get_root_region(MapperContext ctx,
                                         const LogicalPartition &handle) {
  return get_root_region(ctx, mapper_rt_get_parent_logical_region(ctx, handle));
}

static void create_mappers(Machine machine,
                           Runtime *runtime,
                           const std::set<Processor> &local_procs
) {
  for (
    std::set<Processor>::const_iterator it = local_procs.begin();
    it != local_procs.end();
    it++) {
    runtime->replace_default_mapper(
      new EbbMapper(machine, *it, "ebb_mapper"), *it
    );
  }
}

void register_ebb_mappers() {
  HighLevelRuntime::set_registration_callback(create_mappers);
}
