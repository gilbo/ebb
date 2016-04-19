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

#include "shim_mapper.h"
#include "ebb_mapper.h"
#include "serialize.h"

using namespace LegionRuntime::HighLevel;

// legion logger
LegionRuntime::Logger::Category log_mapper("mapper");

// message types
enum MapperMessageType {
  MAPPER_RECORD_FIELD,
  MAPPER_TOTAL_MESSAGES
};

class EbbMapper : public ShimMapper {
public:
  EbbMapper(Machine machine, HighLevelRuntime *rt, Processor local);
  virtual void select_task_options(Task *task);
  virtual bool map_task(Task *task);
  virtual bool map_inline(Inline *inline_operation);
  virtual void notify_mapping_failed(const Mappable *mappable);
  virtual void handle_message(Processor source,
                              const void *message, size_t length);
private:
  // active fields for a logical region
  std::map<std::string, std::set<FieldID> >   active_fields;
  std::vector<Processor>                      all_procs;
  int                                         n_nodes;
  int                                         per_node;
  // get logical region corresponding to a region requirement
  LogicalRegion get_logical_region(const RegionRequirement &req);
  LogicalRegion get_root_region(const LogicalRegion &handle);
  LogicalRegion get_root_region(const LogicalPartition &handle);
};  // class EbbMapper

// EbbMapper constructor
EbbMapper::EbbMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : ShimMapper(machine, rt, local)
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

LogicalRegion EbbMapper::get_logical_region(const RegionRequirement &req) {
  LogicalRegion root;
  if (req.handle_type == SINGULAR || req.handle_type == REG_PROJECTION) {
    root = get_root_region(req.region);
  } else {
    assert(req.handle_type == PART_PROJECTION);
    root = get_root_region(req.partition);
  }
  return root;
}

LogicalRegion EbbMapper::get_root_region(const LogicalRegion &handle) {
  if (has_parent_logical_partition(handle)) {
    return get_root_region(get_parent_logical_partition(handle));
  }
  return handle;
}

LogicalRegion EbbMapper::get_root_region(const LogicalPartition &handle) {
  return get_root_region(get_parent_logical_region(handle));
}

void EbbMapper::select_task_options(Task *task) {
  ShimMapper::select_task_options(task);
  if (!task->regions.empty() && task->regions[0].handle_type == SINGULAR) {
    Color index = get_logical_region_color(task->regions[0].region);
    int proc_off = (int(index) / n_nodes)%per_node;
    int node_off = int(index) % n_nodes;
    task->target_proc =
      all_procs[(node_off*per_node + proc_off)%all_procs.size()];
  }
}

bool EbbMapper::map_task(Task *task) {
  bool success = ShimMapper::map_task(task);

  // add additional fields to region requirements
  std::vector<RegionRequirement> &regions = task->regions;
  for (std::vector<RegionRequirement>::iterator it = regions.begin();
        it != regions.end(); it++) {
    RegionRequirement &req = *it;
    if (!req.redop) {
      LogicalRegion root = get_logical_region(req);
      const char *name_c;
      runtime->retrieve_name(root, name_c);
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
      req.additional_fields.insert(additional.begin(),
                                   additional.end());
    }
  }

  return success;
}

bool EbbMapper::map_inline(Inline *inline_operation) {
  bool success = ShimMapper::map_inline(inline_operation);

  // determine logical region and fields
  RegionRequirement &req = inline_operation->requirement;
  LogicalRegion root = get_logical_region(req);
  const char *name_c;
  runtime->retrieve_name(root, name_c);
  std::string name(name_c);

  // broadcast this information
  Realm::Serialization::DynamicBufferSerializer buffer(name.size() + 16);
  buffer << (int)MAPPER_RECORD_FIELD;
  buffer << name;
  buffer << req.privilege_fields;
  // broadcast_message(buffer.get_buffer(), buffer.bytes_used());

  // add information to local map
  active_fields[name].insert(req.privilege_fields.begin(),
                             req.privilege_fields.end());
  return success;
}

void EbbMapper::notify_mapping_failed(const Mappable *mappable) {
  switch (mappable->get_mappable_kind()) {
  case Mappable::TASK_MAPPABLE:
    {
      log_mapper.warning("mapping failed on task");
      break;
    }
  case Mappable::COPY_MAPPABLE:
    {
      log_mapper.warning("mapping failed on copy");
      break;
    }
  case Mappable::INLINE_MAPPABLE:
    {
      Inline *_inline = mappable->as_mappable_inline();
      RegionRequirement &req = _inline->requirement;
      LogicalRegion region = req.region;
      log_mapper.warning(
        "mapping %s on inline region (%d,%d,%d) memory " IDFMT,
        (req.mapping_failed ? "failed" : "succeeded"),
        region.get_index_space().get_id(),
        region.get_field_space().get_id(),
        region.get_tree_id(),
        req.selected_memory.id);
      break;
    }
  case Mappable::ACQUIRE_MAPPABLE:
    {
      log_mapper.warning("mapping failed on acquire");
      break;
    }
  case Mappable::RELEASE_MAPPABLE:
    {
      log_mapper.warning("mapping failed on release");
      break;
    }
  }
  assert(0 && "mapping failed");
}

static void create_mappers(Machine machine,
                           HighLevelRuntime *runtime,
                           const std::set<Processor> &local_procs
) {
  for (
    std::set<Processor>::const_iterator it = local_procs.begin();
    it != local_procs.end();
    it++) {
    runtime->replace_default_mapper(
      new EbbMapper(machine, runtime, *it), *it
    );
  }
}

void EbbMapper::handle_message(Processor source,
                               const void *message, size_t length) {
  Realm::Serialization::FixedBufferDeserializer buffer(message, length);
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

void register_ebb_mappers() {
  HighLevelRuntime::set_registration_callback(create_mappers);
}
