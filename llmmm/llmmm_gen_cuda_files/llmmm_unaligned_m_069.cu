#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_69_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<69> mm_instantiator;

public:
  UnalignedM_69_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_69_MMInstantiatorWrapper__;

}  // namespace LLMMM
